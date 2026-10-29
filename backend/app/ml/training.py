import math
import json
import shutil
import time
from uuid import uuid4
import numpy as np
import torch
from safetensors.torch import load_file
from .. import store
from ..config import DATA
from dataclasses import replace
from .network import device
from .sequence_data import prepare
from . import style
from .. import difficulty
from .sequence_batches import batches, tasks_for, ExampleWorkers
from .phrase_model import PhraseMapper
from .phrase_losses import parts, objective, LossMeter
from .phrase_metrics import ValidationMetrics
from .training_setup import configuration, split_maps, metadata_for, dataset_signature, publish
from .architecture import ModelConfig
from .checkpoints import save_weights, save_training, restore, warm_start
from .rollout_evaluation import evaluate_suite
from .training_control import TrainingControl, context_probability
from .quality_tracking import grouped_losses, update_quality_bests, training_summary
from .training_runtime import TrainingRuntime
from .batch_loading import device_batches
from .cuda_graph import CudaTrainingGraph


def tensors(batch, selected_device):
    return {key: torch.from_numpy(value).to(selected_device) for key, value in batch.items()}


def evaluate(model, maps, batch_size, selected_device, context, weight, runtime=None, workers=None):
    runtime = runtime or TrainingRuntime(selected_device, "float32")
    meter, quality = LossMeter(), ValidationMetrics()
    model.eval()
    source = batches(maps, batch_size, np.random.default_rng(0), False, workers=workers)
    with torch.no_grad(), device_batches(source, selected_device) as loaded:
        for batch in loaded:
            context.check()
            output = runtime.forward(model, batch)
            meter.add(parts(model, output, batch, weight))
            quality.add(output, batch)
    return meter.value(), {**quality.result(), "loss_terms": meter.values()}


def train(settings, context):
    selected_device = device(settings["device"])
    settings = {
        "model_size": "standard",
        "gradient_accumulation": 4,
        "weight_decay": 0.01,
        "patience": 12,
        "resume_model_id": None,
        "init_model_id": None,
        "mapper_min_maps": 10,
        "precision": "auto",
        "cuda_graphs": True,
        "context_robustness": False,
        "rollout_validation": True,
        **settings,
    }
    source = (
        store.get("model", settings["resume_model_id"]) if settings["resume_model_id"] else None
    )
    if source and ModelConfig.from_dict(source["model_config"]).style_version < 2:
        raise ValueError(
            "Dieses Modell stammt aus der Zeit vor Sterne-, Stil- und Mapper-Steuerung und kann "
            "nicht fortgesetzt werden. Wähle es stattdessen als Startpunkt (Gewichte übernehmen)."
        )
    base = (
        store.get("model", settings["init_model_id"])
        if settings["init_model_id"] and not source
        else None
    )
    if source:
        settings["seed"] = source["settings"]["seed"]
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    issues, star_warning = [], None
    if difficulty.available():
        try:
            records, _ = difficulty.annotate_dataset(settings["dataset_id"], context)
        except ValueError as error:
            records = store.get("dataset", settings["dataset_id"])["maps"]
            star_warning = f"Sterne konnten nicht berechnet werden: {error}"
    else:
        records = store.get("dataset", settings["dataset_id"])["maps"]
        star_warning = (
            "Sterne-Rechner fehlt (scripts/setup.ps1 erneut ausführen): Das Modell lernt ohne "
            "Star Rating und kann später keine Ziel-Sterne ansteuern."
        )
    prepared = prepare(records, context, issues)
    known = style.known_mappers(records)
    for item in prepared:
        item["mapper_name"] = style.resolve_mapper(
            item.get("creator"), item.get("mapper_candidate"), known
        )
    split = split_maps(prepared, settings["seed"])
    training, validation, test = split
    if not training:
        raise ValueError("No training songs remain after splitting")
    if source and source.get("dataset_signature") != dataset_signature(prepared):
        raise ValueError("Dataset differs from the saved training run; start a new model")
    if source:
        vocabulary = list(source.get("mapper_vocab", []))
    else:
        vocabulary = list(base.get("mapper_vocab", [])) if base else []
        vocabulary += [
            name
            for name in style.mapper_vocabulary(
                [item["mapper_name"] for item in training], settings["mapper_min_maps"]
            )
            if name not in vocabulary
        ][: 4095 - len(vocabulary)]
    slots = {name: index + 1 for index, name in enumerate(vocabulary)}
    for item in prepared:
        item["mapper"] = slots.get(item["mapper_name"], 0)
    if source:
        cfg = ModelConfig.from_dict(source["model_config"])
    elif base:
        # Keep the base model's dimensions so its audio and decoder weights can be reused.
        cfg = replace(
            ModelConfig.from_dict(base["model_config"]),
            style_version=2,
            mapper_slots=len(vocabulary) + 1,
        )
    else:
        cfg = replace(configuration(settings["model_size"]), mapper_slots=len(vocabulary) + 1)
    model = PhraseMapper(cfg).to(selected_device)
    transferred = warm_start(model, base["id"]) if base else None
    runtime = TrainingRuntime(selected_device, settings["precision"])
    optimizer = runtime.optimizer(model, settings["learning_rate"], settings["weight_decay"])
    start_epoch, step, state = 0, 0, {}
    if source:
        _, state = restore(model, optimizer, source["id"], settings["dataset_id"])
        start_epoch, step = state["completed_epochs"], state["optimizer_steps"]
    frame_count = sum(item["labels"][:, 4].sum() for item in training)
    hits = sum(item["labels"][:, 0].sum() for item in training)
    weight = float(np.clip(frame_count / max(1, 2 * hits) - 1, 1, 40))
    identifier = uuid4().hex
    directory = DATA / "models" / identifier
    directory.mkdir()
    metadata = metadata_for(identifier, settings, cfg, prepared, split, issues)
    metadata.update(
        style_version=cfg.style_version,
        style_features=list(style.KEYS),
        mapper_vocab=vocabulary,
        mapper_profiles=style.mapper_profiles(prepared, vocabulary),
        star_profiles=style.star_profiles(training + validation),
        rated_maps=int(
            sum(
                item["style_mask"][style.INDEX["stars"]] > 0
                for item in prepared
                if "style_mask" in item
            )
        ),
        difficulty_calculator=next(
            (
                record["difficulty_attributes"].get("calculator")
                for record in records
                if record.get("difficulty_attributes")
            ),
            None,
        ),
    )
    if star_warning:
        metadata["warnings"].append(star_warning)
    if transferred:
        metadata["warm_start"] = {"model_id": base["id"], "name": base.get("name"), **transferred}
    # Keep interrupted/cancelled training history attributable after checkpoint recovery.
    metadata["job_id"] = getattr(context, "id", None)
    metadata["execution"] = {
        "precision": runtime.precision,
        "cuda_sdpa": runtime.cuda,
        "fused_optimizer": runtime.cuda,
    }
    metadata["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    metadata["onset_positive_weight"] = weight
    metadata["training_policy"] = "plateau-context-v1"
    metadata["selection_rule"] = "minimum_total_validation_loss"
    metrics, best = [], math.inf
    if source:
        metrics = list(source["metrics"])
        best = source["best_loss"] if source["best_loss"] is not None else math.inf
        metadata["best_epoch"] = source["best_epoch"]
        metadata["quality_bests"] = dict(source.get("quality_bests", {}))
        shutil.copyfile(
            DATA / "models" / source["id"] / "model.safetensors", directory / "model.safetensors"
        )
    tasks = tasks_for(training)
    batch_count = math.ceil(len(tasks) / settings["batch_size"])
    accumulation = settings["gradient_accumulation"]
    use_graph = runtime.cuda and settings["cuda_graphs"] and accumulation == 1
    captured = None
    metadata["execution"]["cuda_graphs"] = use_graph
    warmup = min(200, max(1, math.ceil(batch_count / accumulation))) if not source else 0
    control = TrainingControl(
        settings["learning_rate"], settings["patience"], warmup, best, state.get("training_control")
    )
    # Old checkpoints have no controller state: start from their last measured rate.
    if source and not state.get("training_control") and metrics:
        control.lr = min(control.lr, metrics[-1].get("learning_rate", control.lr))
    local_updates, completed_epochs, published, evaluating_best = 0, start_epoch, False, False
    # Batches are built in worker processes so the GPU is not left waiting on NumPy.
    example_workers = ExampleWorkers()
    metadata["execution"]["batch_workers"] = example_workers.workers

    def checkpoint():
        save_training(
            model,
            optimizer,
            directory,
            {
                "completed_epochs": completed_epochs,
                "optimizer_steps": step,
                "training_control": control.state(),
            },
        )
        if not math.isfinite(best):
            save_weights(model, directory / "model.safetensors")
        recovery = {
            **metadata,
            "metrics": metrics,
            "best_loss": best if math.isfinite(best) else None,
            "completed_epochs": completed_epochs,
            "training_status": "interrupted",
        }
        temporary = directory / "recovery.tmp"
        temporary.write_text(json.dumps(recovery, allow_nan=False), encoding="utf-8")
        temporary.replace(directory / "recovery.json")

    try:
        for local_epoch in range(settings["epochs"]):
            epoch = start_epoch + local_epoch + 1
            meter = LossMeter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            start = runtime.begin_measurement()
            examples, epoch_updates, last_report = 0, step, 0.0
            finite_loss = torch.ones((), dtype=torch.bool, device=selected_device)
            source_batches = batches(
                training, settings["batch_size"], rng, True, tasks, workers=example_workers
            )
            with device_batches(source_batches, selected_device) as loaded:
                for batch_index, batch in enumerate(loaded):
                    context.check()
                    examples += batch["audio"].shape[0]
                    probability = context_probability(epoch, settings["context_robustness"])
                    if probability > 0:
                        batch["context_probability"] = torch.full(
                            (1,), probability, device=selected_device
                        )
                    # Recapture once when the auxiliary decoder becomes active after warmup.
                    if captured is not None and batch.keys() != captured.static.keys():
                        captured = None
                    if use_graph and captured is None:
                        context.report(
                            0.15, "CUDA-Rechenablauf wird einmalig vorbereitet", metrics=metrics
                        )
                        captured = CudaTrainingGraph(model, runtime, batch, weight)
                    graph_batch = captured is not None and captured.matches(batch)
                    if graph_batch:
                        loss, terms = captured.replay(batch)
                    else:
                        output = runtime.forward(model, batch)
                        terms = parts(model, output, batch, weight)
                        loss = objective(terms)
                    finite_loss = finite_loss & torch.isfinite(loss.detach())
                    # Normalize the final partial accumulation group as well.
                    group_size = min(
                        accumulation, batch_count - (batch_index // accumulation) * accumulation
                    )
                    if not graph_batch:
                        (loss / group_size).backward()
                    meter.add(terms)
                    if (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_count:
                        runtime.clip_and_check(model, finite_loss)
                        finite_loss = torch.ones((), dtype=torch.bool, device=selected_device)
                        for group in optimizer.param_groups:
                            group["lr"] = control.step_rate()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        step += 1
                        local_updates += 1
                        control.updates += 1
                        if local_updates == 1 or local_updates % 500 == 0:
                            checkpoint()
                    progress = (local_epoch + (batch_index + 1) / batch_count) / settings["epochs"]
                    if time.monotonic() - last_report >= 1 or batch_index + 1 == batch_count:
                        last_report = time.monotonic()
                        context.report(
                            0.15 + 0.8 * progress,
                            f"Epoche {epoch} · Batch {batch_index+1}/{batch_count}",
                            metrics=metrics,
                            warnings=metadata["warnings"],
                        )
            performance = runtime.measurement(start, examples, step - epoch_updates)
            validation_loss, quality = evaluate(
                model,
                validation,
                settings["batch_size"],
                selected_device,
                context,
                weight,
                workers=example_workers,
            )
            train_terms = meter.values()
            # Auxiliary context/plan losses stay separate from comparable base curves.
            train_loss = meter.value()
            metric = {
                "epoch": epoch,
                "performance": performance,
                "train_loss": train_loss,
                "validation_loss": validation_loss if validation else None,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_terms": train_terms,
                "context_probability": context_probability(epoch, settings["context_robustness"]),
                "train_groups": grouped_losses(train_terms),
                "validation_groups": grouped_losses(quality["loss_terms"]) if validation else {},
                **(quality if validation else {}),
            }
            metrics.append(metric)
            score = validation_loss if validation else train_loss
            improved, reduced, stop = control.observe(score, bool(validation))
            best = control.best
            if improved:
                metadata["best_epoch"] = epoch
                save_weights(model, directory / "model.safetensors")
            metric.update(
                next_learning_rate=control.lr,
                learning_rate_reduced=reduced,
                best_epoch=metadata["best_epoch"],
            )
            update_quality_bests(metadata, metric)
            completed_epochs = epoch
            metadata["training_summary"] = training_summary(metadata, control, epoch, stop)
            checkpoint()
            if (
                validation
                and settings["rollout_validation"]
                and (epoch == 1 or epoch % 5 == 0 or stop or local_epoch + 1 == settings["epochs"])
            ):
                metric["free_validation"] = evaluate_suite(
                    model,
                    validation,
                    selected_device,
                    context,
                    progress=0.15 + 0.8 * (local_epoch + 1) / settings["epochs"],
                )
                checkpoint()
            context.report(
                0.15 + 0.8 * (local_epoch + 1) / settings["epochs"],
                f"Epoche {epoch} abgeschlossen",
                metrics=metrics,
                warnings=metadata["warnings"],
                training_summary=metadata["training_summary"],
            )
            if stop:
                metadata["warnings"].append(
                    "Frühstopp: der Prüfverlust hat sich nicht weiter verbessert."
                )
                break
        evaluating_best = True
        model.load_state_dict(load_file(str(directory / "model.safetensors")))
        if test:
            loss, quality = evaluate(
                model,
                test,
                settings["batch_size"],
                selected_device,
                context,
                weight,
                workers=example_workers,
            )
            metadata["heldout_test"] = {"loss": loss, **quality}
        if test or validation:
            suite = evaluate_suite(
                model, test or validation, selected_device, context, include_full_song=True
            )
            metadata["free_generation_suite"] = {
                **suite,
                "split": "test" if test else "validation",
                "epoch": metadata["best_epoch"],
            }
            if suite["results"]:
                metadata["free_generation"] = suite["results"][0]
        metadata.update(
            metrics=metrics,
            best_loss=best,
            training_status="completed",
            completed_epochs=completed_epochs,
        )
        if metadata.get("training_summary", {}).get("stop_reason") is None:
            metadata["training_summary"]["stop_reason"] = "epoch_limit"
        context.report(
            0.99,
            f"Gespeichert: Epoche {metadata['best_epoch']}",
            metrics=metrics,
            warnings=metadata["warnings"],
            training_summary=metadata["training_summary"],
        )
        context.check()
        publish(directory, metadata)
        published = True
        return {"model_id": identifier}
    except Exception as error:
        if local_updates:
            try:
                if not evaluating_best:
                    checkpoint()
                if not (directory / "model.safetensors").exists():
                    save_weights(model, directory / "model.safetensors")
                    metadata["best_epoch"] = completed_epochs
                metadata.update(
                    metrics=metrics,
                    best_loss=best if math.isfinite(best) else None,
                    training_status="interrupted",
                    completed_epochs=completed_epochs,
                )
                metadata["warnings"].append(
                    "Lauf unterbrochen. Letzter Trainingsstand kann fortgesetzt werden; eine angebrochene Epoche beginnt erneut."
                )
                publish(directory, metadata)
                published = True
            except Exception:
                # Preserve an already written, finite checkpoint even if saving the latest state fails.
                if (directory / "training.safetensors").exists() and (
                    directory / "model.safetensors"
                ).exists():
                    metadata.update(
                        metrics=metrics,
                        best_loss=best if math.isfinite(best) else None,
                        training_status="interrupted",
                        completed_epochs=completed_epochs,
                    )
                    publish(directory, metadata)
                    published = True
        raise error
    finally:
        example_workers.close()
        if not published:
            shutil.rmtree(directory)

import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { MapItem } from "../types";

export interface MapFilters {
  search: string;
  minimum: string;
  maximum: string;
  perSong: "all" | "easiest" | "middle" | "hardest";
}

export function useLibraryMaps(
  revision: string,
  onError: (message: string) => void,
) {
  const [filters, setFilters] = useState<MapFilters>({
    search: "",
    minimum: "",
    maximum: "",
    perSong: "all",
  });
  const [offset, setOffset] = useState(0);
  const [items, setItems] = useState<MapItem[]>([]);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [selecting, setSelecting] = useState(false);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const generation = useRef(0);
  const minimum = filters.minimum === "" ? null : Number(filters.minimum);
  const maximum = filters.maximum === "" ? null : Number(filters.maximum);
  const invalid =
    [minimum, maximum].some(
      (value) => value !== null && (!Number.isFinite(value) || value < 0),
    ) ||
    (minimum !== null && maximum !== null && minimum > maximum);
  const query = new URLSearchParams({ q: filters.search });
  query.set("per_song", filters.perSong);
  if (minimum !== null) query.set("stars_min", String(minimum));
  if (maximum !== null) query.set("stars_max", String(maximum));
  const queryString = query.toString();

  useEffect(
    () => () => {
      generation.current += 1;
    },
    [],
  );
  // Filter changes start from an empty table; a refresh of the same query
  // keeps the rows on screen (dimmed) until the answer arrives.
  const view = `${queryString}&offset=${offset}`;
  const shown = useRef("");
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    if (shown.current !== view) {
      setItems([]);
      setTotal(0);
    }
    if (invalid) {
      setLoading(false);
      return;
    }
    const timer = setTimeout(() => {
      api<{ items: MapItem[]; total: number }>(
        `/maps?${queryString}&offset=${offset}&limit=100`,
      )
        .then((result) => {
          if (active) {
            shown.current = view;
            setItems(result.items);
            setTotal(result.total);
          }
        })
        .catch((reason) => {
          if (active) {
            shown.current = "";
            setItems([]);
            setTotal(0);
            setError((reason as Error).message);
          }
        })
        .finally(() => {
          if (active) setLoading(false);
        });
    }, 180);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [queryString, offset, view, revision, invalid, attempt]);

  function updateFilters(change: Partial<MapFilters>) {
    const next = { ...filters, ...change };
    if (
      Object.keys(next).every(
        (key) =>
          next[key as keyof MapFilters] === filters[key as keyof MapFilters],
      )
    )
      return;
    generation.current += 1;
    setFilters(next);
    setOffset(0);
    setSelected(new Set());
  }

  async function selectAll() {
    const current = generation.current;
    setSelecting(true);
    try {
      const result = await api<{ ids: string[]; total: number }>(
        `/maps/ids?${queryString}`,
      );
      if (current === generation.current) {
        setSelected(new Set(result.ids));
        setTotal(result.total);
      }
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setSelecting(false);
    }
  }

  function toggle(identifier: string) {
    generation.current += 1;
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(identifier)) next.delete(identifier);
      else next.add(identifier);
      return next;
    });
  }

  function togglePage(checked: boolean) {
    generation.current += 1;
    setSelected((current) => {
      const next = new Set(current);
      items.forEach((item) =>
        checked ? next.add(item.id) : next.delete(item.id),
      );
      return next;
    });
  }

  function clearSelection() {
    generation.current += 1;
    setSelected(new Set());
  }

  return {
    filters,
    updateFilters,
    invalid,
    offset,
    setOffset,
    items,
    total,
    selected,
    togglePage,
    loading,
    selecting,
    selectAll,
    toggle,
    clearSelection,
    error,
    retry: () => setAttempt((value) => value + 1),
  };
}

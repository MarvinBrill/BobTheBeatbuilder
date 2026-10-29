import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { AlertCircle, Check, Info, X } from "lucide-react";

export type ToastTone = "error" | "success" | "info";
export interface ToastAction {
  label: string;
  /** A link (downloads stay links so the browser handles them). */
  href?: string;
  run?: () => void;
}
export interface ToastInput {
  tone: ToastTone;
  title: string;
  message?: string;
  action?: ToastAction;
  /** Same key replaces the earlier toast instead of stacking a copy. */
  key?: string;
}
interface Toast extends ToastInput {
  id: number;
  leaving?: boolean;
}

const DWELL = { info: 4500, success: 6000 };
/** A toast you might act on gets longer to read. */
const DWELL_WITH_ACTION = 7000;
const EXIT = 180;
const LIMIT = 2;

const ToastContext = createContext<{
  notify: (toast: ToastInput) => void;
  dismiss: (key: string) => void;
}>({ notify: () => {}, dismiss: () => {} });

export function useToast() {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const counter = useRef(0);

  const close = useCallback((id: number) => {
    setToasts((current) =>
      current.map((toast) =>
        toast.id === id ? { ...toast, leaving: true } : toast,
      ),
    );
    setTimeout(
      () => setToasts((current) => current.filter((toast) => toast.id !== id)),
      EXIT,
    );
  }, []);

  const notify = useCallback((input: ToastInput) => {
    const key = input.key ?? `${input.tone}:${input.title}:${input.message}`;
    counter.current += 1;
    const toast = { ...input, key, id: counter.current };
    setToasts((current) =>
      [...current.filter((item) => item.key !== key), toast].slice(-LIMIT),
    );
  }, []);

  const dismiss = useCallback(
    (key: string) =>
      setToasts((current) => current.filter((toast) => toast.key !== key)),
    [],
  );

  const value = useMemo(() => ({ notify, dismiss }), [notify, dismiss]);
  return (
    <ToastContext.Provider value={value}>
      {children}
      <section
        className="toast-region"
        aria-label="Benachrichtigungen"
        aria-live="polite"
      >
        {toasts.map((toast) => (
          <ToastCard key={toast.id} toast={toast} onClose={close} />
        ))}
      </section>
    </ToastContext.Provider>
  );
}

function ToastCard({
  toast,
  onClose,
}: {
  toast: Toast;
  onClose: (id: number) => void;
}) {
  const [paused, setPaused] = useState(false);
  // Remaining dwell time; pausing keeps what is left instead of restarting.
  const remaining = useRef(
    toast.tone === "error"
      ? Infinity
      : toast.action
        ? DWELL_WITH_ACTION
        : DWELL[toast.tone],
  );
  useEffect(() => {
    if (!Number.isFinite(remaining.current) || paused || toast.leaving) return;
    const started = Date.now();
    const timer = setTimeout(() => onClose(toast.id), remaining.current);
    return () => {
      clearTimeout(timer);
      remaining.current = Math.max(0, remaining.current - (Date.now() - started));
    };
  }, [paused, toast.id, toast.leaving, onClose]);
  const Icon =
    toast.tone === "error" ? AlertCircle : toast.tone === "success" ? Check : Info;
  const action = toast.action;
  return (
    <div
      className={`toast ${toast.tone}${toast.leaving ? " leaving" : ""}`}
      role={toast.tone === "error" ? "alert" : undefined}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <Icon size={18} className="toast-icon" aria-hidden="true" />
      <div className="toast-body">
        <b>{toast.title}</b>
        {toast.message && <p>{toast.message}</p>}
        {action &&
          (action.href ? (
            <a
              className="toast-action"
              href={action.href}
              onClick={() => onClose(toast.id)}
            >
              {action.label}
            </a>
          ) : (
            <button
              type="button"
              className="toast-action"
              onClick={() => {
                action.run?.();
                onClose(toast.id);
              }}
            >
              {action.label}
            </button>
          ))}
      </div>
      <button
        type="button"
        className="icon-button"
        aria-label="Benachrichtigung schließen"
        onClick={() => onClose(toast.id)}
      >
        <X size={16} />
      </button>
    </div>
  );
}

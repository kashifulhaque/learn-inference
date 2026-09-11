import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";

type Props = {
  /** "row" places the panes side by side, "column" stacks them. */
  direction?: "row" | "column";
  /** Where the size the reader drags to is remembered. */
  storageKey: string;
  /** Percentage of the container the first pane takes before any drag. */
  initial?: number;
  min?: number;
  max?: number;
  first: ReactNode;
  second: ReactNode;
  className?: string;
  label?: string;
};

function remembered(key: string, fallback: number): number {
  try {
    const raw = window.localStorage.getItem(key);
    const value = raw === null ? NaN : Number(raw);
    return Number.isFinite(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

/**
 * Two panes with a divider the reader can drag, the way an IDE splits an editor
 * from its console. The split is a percentage so it survives a window resize,
 * and it is remembered per `storageKey` so a chapter opens the way it was left.
 */
export default function SplitPane({
  direction = "row",
  storageKey,
  initial = 50,
  min = 20,
  max = 80,
  first,
  second,
  className = "",
  label = "Resize the panes",
}: Props) {
  const [size, setSize] = useState(() => remembered(storageKey, initial));
  const [dragging, setDragging] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const row = direction === "row";

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, String(Math.round(size * 10) / 10));
    } catch {
      // A browser with storage switched off still gets a working split.
    }
  }, [storageKey, size]);

  const moveTo = useCallback(
    (clientX: number, clientY: number) => {
      const box = container.current?.getBoundingClientRect();
      if (!box) return;
      const fraction = row
        ? (clientX - box.left) / box.width
        : (clientY - box.top) / box.height;
      setSize(Math.min(max, Math.max(min, fraction * 100)));
    },
    [row, min, max],
  );

  useEffect(() => {
    if (!dragging) return;
    const onMove = (event: PointerEvent) => {
      event.preventDefault();
      moveTo(event.clientX, event.clientY);
    };
    const stop = () => setDragging(false);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
    // While dragging, the pointer regularly leaves the divider. Suppressing
    // selection and hover keeps the drag from painting text blue on the way.
    const previous = document.body.style.userSelect;
    document.body.style.userSelect = "none";
    document.body.style.cursor = row ? "col-resize" : "row-resize";
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
      document.body.style.userSelect = previous;
      document.body.style.cursor = "";
    };
  }, [dragging, moveTo, row]);

  function nudge(delta: number) {
    setSize((current) => Math.min(max, Math.max(min, current + delta)));
  }

  const firstStyle: CSSProperties = row
    ? { width: `${size}%` }
    : { height: `${size}%` };

  return (
    <div
      ref={container}
      className={`flex min-h-0 min-w-0 ${row ? "flex-row" : "flex-col"} ${className}`}
    >
      <div className="flex min-h-0 min-w-0 flex-col" style={firstStyle}>
        {first}
      </div>

      <div
        role="separator"
        aria-orientation={row ? "vertical" : "horizontal"}
        aria-label={label}
        aria-valuenow={Math.round(size)}
        aria-valuemin={min}
        aria-valuemax={max}
        tabIndex={0}
        onPointerDown={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDoubleClick={() => setSize(initial)}
        onKeyDown={(event) => {
          const back = row ? "ArrowLeft" : "ArrowUp";
          const forward = row ? "ArrowRight" : "ArrowDown";
          if (event.key === back) nudge(-2);
          else if (event.key === forward) nudge(2);
          else if (event.key === "Home") setSize(initial);
          else return;
          event.preventDefault();
        }}
        className={`group relative shrink-0 bg-ink-800 transition-colors ${
          row ? "w-px cursor-col-resize" : "h-px cursor-row-resize"
        } ${dragging ? "bg-flame-500" : "hover:bg-flame-500/60"}`}
      >
        {/* The visible line is a hairline; this widens what the pointer hits. */}
        <span
          className={`absolute ${
            row ? "-inset-x-2 inset-y-0" : "-inset-y-2 inset-x-0"
          }`}
          aria-hidden
        />
      </div>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">{second}</div>
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import type { Lecture, LectureSection } from "../lib/api";

/** Seconds as `m:ss`, or `h:mm:ss` past an hour. */
export function clock(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

/** The section playing at `time`, or null in a gap. */
function sectionAt(sections: LectureSection[], time: number): LectureSection | null {
  return sections.find((section) => time >= section.start && time < section.end) ?? null;
}

/** A `##` section of the chapter text, as the reader numbers it. */
export type BookSection = { index: number; id: string; title: string };

/**
 * A chapter's lecture: the video with its captions, and a list of its
 * sections that seeks the video and links each section to the part of the
 * chapter it teaches.
 *
 * The video uses the browser's own controls, which already handle the
 * keyboard, captions, speed, and full screen.
 *
 * The media come from another origin, so the video loads with CORS. The
 * bucket allows it for this site, and without it the browser drops the
 * captions track. The slide thumbnails are plain images and load without it.
 *
 * `book` maps each heading id the chapter renders to its section. A lecture
 * section links to the text only when its `book_anchor` is one of them, and
 * `onRead` scrolls the reader there.
 */
export default function LecturePlayer({
  lecture,
  book,
  onRead,
}: {
  lecture: Lecture;
  book: Map<string, BookSection>;
  onRead: (id: string) => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const list = useRef<HTMLOListElement>(null);
  const items = useRef(new Map<string, HTMLLIElement>());
  const [current, setCurrent] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  const { width, height } = lecture.video;
  const ratio = width && height ? `${width} / ${height}` : "16 / 9";

  // Keep the playing section in view inside the list, without scrolling the
  // chapter around it.
  useEffect(() => {
    const item = current ? items.current.get(current) : undefined;
    const box = list.current;
    if (!item || !box) return;
    const itemRect = item.getBoundingClientRect();
    const boxRect = box.getBoundingClientRect();
    if (itemRect.top < boxRect.top) box.scrollTop -= boxRect.top - itemRect.top;
    else if (itemRect.bottom > boxRect.bottom) box.scrollTop += itemRect.bottom - boxRect.bottom;
  }, [current]);

  function onTime() {
    const node = video.current;
    if (!node) return;
    const id = sectionAt(lecture.sections, node.currentTime)?.id ?? null;
    if (id !== current) setCurrent(id);
  }

  function seek(section: LectureSection) {
    const node = video.current;
    if (!node) return;
    node.currentTime = section.start;
    setCurrent(section.id);
    void node.play().catch(() => undefined);
  }

  return (
    <div>
      <div className="relative overflow-hidden bg-black" style={{ aspectRatio: ratio }}>
        {failed ? (
          <p className="absolute inset-0 flex items-center justify-center px-6 text-center text-[13px] text-white/75">
            The lecture video didn't load. Check your connection, then reload the page.
          </p>
        ) : (
          <video
            ref={video}
            src={lecture.video.src}
            poster={lecture.video.poster ?? undefined}
            crossOrigin="anonymous"
            controls
            playsInline
            preload="metadata"
            aria-label={`Lecture: ${lecture.title}`}
            className="absolute inset-0 h-full w-full object-contain"
            onTimeUpdate={onTime}
            onSeeked={onTime}
            onError={() => setFailed(true)}
          >
            {lecture.captions && (
              <track kind="captions" src={lecture.captions} srcLang="en" label="English" default />
            )}
          </video>
        )}
      </div>

      <ol
        ref={list}
        aria-label="Sections of the lecture"
        className="max-h-80 overflow-y-auto overscroll-contain border-t border-line p-1.5"
      >
        {lecture.sections.map((section, index) => {
          const on = section.id === current;
          const target = section.book_anchor ? book.get(section.book_anchor) : undefined;
          return (
            <li
              key={section.id}
              ref={(node) => {
                if (node) items.current.set(section.id, node);
                else items.current.delete(section.id);
              }}
              className={`flex items-center gap-1 rounded-md transition-colors ${
                on ? "bg-accent/8" : "hover:bg-tint"
              }`}
            >
              <button
                type="button"
                onClick={() => seek(section)}
                disabled={failed}
                aria-current={on ? "true" : undefined}
                aria-label={`${section.title}, at ${clock(section.start)}`}
                className="flex min-w-0 flex-1 items-center gap-3 rounded-md px-1.5 py-1.5 text-left disabled:cursor-default disabled:opacity-50"
              >
                <span className="relative hidden w-20 shrink-0 overflow-hidden rounded border border-line bg-well @sm:block @lg:w-24">
                  <span className="block aspect-video" />
                  {section.image && (
                    <img
                      src={section.image}
                      alt=""
                      loading="lazy"
                      draggable={false}
                      className="absolute inset-0 h-full w-full object-cover"
                    />
                  )}
                </span>
                <span
                  className={`w-5 shrink-0 text-right font-mono text-[11.5px] tabular-nums ${
                    on ? "text-accent" : "text-fg-faint"
                  }`}
                >
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span className={`min-w-0 flex-1 text-[13px] leading-5 ${on ? "text-fg" : "text-fg-muted"}`}>
                  {section.title}
                </span>
                <span className="shrink-0 font-mono text-[11.5px] tabular-nums text-fg-subtle">
                  {clock(section.start)}
                </span>
              </button>
              {target ? (
                <a
                  href={`#${target.id}`}
                  onClick={(event) => {
                    event.preventDefault();
                    onRead(target.id);
                  }}
                  title={`Read “${target.title}” in the chapter`}
                  aria-label={`Read “${target.title}” in the chapter`}
                  className="mr-1 flex h-7 shrink-0 items-center rounded-md px-1.5 font-mono text-[11.5px] tabular-nums text-fg-subtle transition-colors hover:bg-card hover:text-accent"
                >
                  §{String(target.index).padStart(2, "0")}
                </a>
              ) : (
                <span className="mr-1 w-[2.6rem] shrink-0" aria-hidden />
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

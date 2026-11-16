import { useEffect, useMemo, useState } from "react";
import { api, type ChapterMeta, type LectureSummary, type Run } from "../lib/api";
import { chapterNumbers, groupByPart, splitPart } from "../lib/chapters";
import { formatAgo, formatMinutes } from "../lib/format";
import { doneCount, nextChapter, partNumber } from "../lib/shellData";
import { clock } from "../components/LecturePlayer";
import StatusMark from "../components/StatusMark";
import { Loading } from "../components/ui";
import Overview, { OverviewBlock, OverviewList } from "./Overview";

type Props = {
  /** Null until the course list has loaded. */
  chapters: ChapterMeta[] | null;
  progress: Record<string, string>;
  model: string;
  gpu: string;
};

const RECENT_RUNS = 5;

function runOutcome(run: Run): string {
  if (run.passed === true) return "Passed";
  if (run.passed === false) return "Failed";
  const words = run.status.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * The learn-inference overview at "/": the shared Overview page filled from
 * the chapter index, with the lectures and the reader's recent lab runs after
 * the part cards.
 */
export default function Dashboard({ chapters, progress, model, gpu }: Props) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [lectures, setLectures] = useState<LectureSummary[]>([]);

  useEffect(() => {
    api.runs().then((result) => setRuns(result.runs)).catch(() => undefined);
    api.lectures().then((result) => setLectures(result.lectures)).catch(() => undefined);
  }, []);

  const list = useMemo(() => chapters ?? [], [chapters]);
  const numbers = useMemo(() => chapterNumbers(list), [list]);
  const byLab = useMemo(
    () => new Map(list.filter((c) => c.lab).map((c) => [c.lab as string, c])),
    [list],
  );

  if (!chapters) return <Loading label="Loading the course" />;

  const done = doneCount(list, progress);
  const next = nextChapter(list, progress);
  const resuming = next ? Boolean(progress[next.slug]) : false;
  const labs = list.filter((c) => c.lab).length;
  // A lab counts once however many times it passed.
  const labsPassed = new Set(runs.filter((r) => r.passed).map((r) => r.lab)).size;
  const minutes = list.reduce((sum, c) => sum + (c.minutes ?? 0), 0);
  const minutesLeft = list
    .filter((c) => progress[c.slug] !== "done")
    .reduce((sum, c) => sum + (c.minutes ?? 0), 0);
  const recent = [...runs].sort((a, b) => b.started_at - a.started_at).slice(0, RECENT_RUNS);
  const parts = groupByPart(list);

  const progressDetail = [
    labs ? `${labsPassed} of ${labs} labs passed` : "",
    runs.length ? `${runs.length} lab run${runs.length === 1 ? "" : "s"}` : "",
    minutesLeft ? `${formatMinutes(minutesLeft)} of reading left` : "Nothing left to read",
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <Overview
      eyebrow="Field notes · learn/inference"
      pill={`${list.length} chapters · ${labs} labs`}
      title={
        <>
          Build an <em>inference engine</em>
        </>
      }
      intro={
        <>
          From a safetensors file to a serving engine: CUDA kernels, a paged cache, continuous
          batching, and hybrid attention for <code>{model || "the target model"}</code>
          {gpu ? `. The labs run on an ${gpu}.` : "."}
        </>
      }
      primary={
        next
          ? { href: `/c/${next.slug}`, label: resuming ? `Continue: ${next.title}` : "Start the course" }
          : list[0]
            ? { href: `/c/${list[0].slug}`, label: "Read it again" }
            : undefined
      }
      secondary={[{ href: "/compute", label: "Compute" }]}
      next={
        next
          ? {
              href: `/c/${next.slug}`,
              kicker: `${resuming ? "Pick up where you left off" : "Start here"} · Chapter ${numbers.get(next.slug) ?? ""}`,
              title: next.title,
              summary: next.summary,
              action: resuming ? "Continue" : "Start chapter",
            }
          : undefined
      }
      progress={{ done, total: list.length, label: "Chapters done", detail: progressDetail }}
      facts={[
        { icon: "book", value: String(list.length), label: "chapters" },
        { icon: "layers", value: String(parts.length), label: "parts" },
        ...(labs > 0 ? [{ icon: "flask" as const, value: String(labs), label: "labs with checks" }] : []),
        ...(minutes > 0 ? [{ icon: "clock" as const, value: formatMinutes(minutes), label: "of reading" }] : []),
      ]}
      path={{
        eyebrow: "The learning path",
        title: "The course, part by part",
        aside: `${parts.length} parts · ${list.length} chapters`,
      }}
      sections={parts.map(([part, items], index) => {
        const partMinutes = items.reduce((sum, c) => sum + (c.minutes ?? 0), 0);
        const first = items[0];
        const last = items[items.length - 1];
        return {
          id: part,
          number: partNumber(part, index),
          title: splitPart(part).name,
          meta: [
            first && last && first !== last
              ? `Chapters ${numbers.get(first.slug)} to ${numbers.get(last.slug)}`
              : `${items.length} chapter`,
            partMinutes > 0 ? formatMinutes(partMinutes) : "",
          ]
            .filter(Boolean)
            .join(" · "),
          lessons: items.map((chapter) => ({
            id: chapter.slug,
            href: `/c/${chapter.slug}`,
            title: chapter.title,
            number: numbers.get(chapter.slug),
            done: progress[chapter.slug] === "done",
            tag: chapter.lab ? (chapter.gpu ? "GPU lab" : "Lab") : undefined,
          })),
        };
      })}
    >
      {lectures.length > 0 && (
        <OverviewBlock
          eyebrow="Watch"
          title="Lectures"
          aside={`${lectures.length} of ${list.length} chapters`}
          id="sh-lectures-heading"
        >
          <OverviewList
            items={lectures.map((lecture) => ({
              id: lecture.slug,
              href: `/c/${lecture.slug}`,
              kicker: `Chapter ${numbers.get(lecture.slug) ?? ""} · ${clock(lecture.duration)} · ${lecture.sections} slides`,
              title: lecture.chapter_title,
              trailing: <StatusMark state={progress[lecture.slug]} />,
            }))}
          />
        </OverviewBlock>
      )}

      {recent.length > 0 && (
        <OverviewBlock
          eyebrow="Activity"
          title="Recent lab runs"
          aside={`${runs.length} in total`}
          id="sh-runs-heading"
        >
          <OverviewList
            items={recent.map((run) => {
              const chapter = byLab.get(run.lab);
              const seconds =
                run.finished_at !== null ? Math.round(run.finished_at - run.started_at) : null;
              return {
                id: run.id,
                href: chapter ? `/c/${chapter.slug}` : "/",
                kicker: [formatAgo(run.started_at), run.provider, seconds !== null ? `${seconds}s` : ""]
                  .filter(Boolean)
                  .join(" · "),
                title: chapter ? chapter.title : run.lab,
                detail: runOutcome(run),
                trailing: (
                  <span
                    className={`size-1.5 shrink-0 rounded-full ${
                      run.passed === true ? "bg-ok" : run.passed === false ? "bg-bad" : "bg-fg-faint"
                    }`}
                    role="img"
                    aria-label={runOutcome(run)}
                  />
                ),
              };
            })}
          />
        </OverviewBlock>
      )}
    </Overview>
  );
}

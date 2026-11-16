/**
 * Builds the shared shell components' props from the learn-inference chapter
 * index. This file is course-specific: each course writes its own version.
 */

import { useEffect, useMemo, useState } from "react";

import type { SearchEntry } from "../components/SearchDialog";
import { api, type ChapterMeta } from "./api";
import { chapterNumbers, splitPart } from "./chapters";

/** Strips the Markdown and the maths a heading can carry, for display. */
function plain(text: string): string {
  return text
    .replace(/\$([^$]+)\$/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*?([^*]+)\*\*?/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/\\[a-zA-Z]+/g, "")
    .replace(/[{}]/g, "")
    .trim();
}

type Heading = { level: 2 | 3; text: string };

/**
 * The `##` and `###` headings of a chapter body, outside code fences. The
 * reader anchors `##` sections only, numbered the way rehypeSectionize
 * numbers them, so `#s-<n>` lands on the same section.
 */
function headings(body: string): Heading[] {
  const found: Heading[] = [];
  let fence: string | null = null;
  for (const line of body.split("\n")) {
    const opener = /^\s*(`{3,}|~{3,})/.exec(line);
    if (opener) {
      if (fence === null) fence = opener[1][0];
      else if (opener[1][0] === fence) fence = null;
      continue;
    }
    if (fence !== null) continue;
    const match = /^(#{2,3})\s+(.*)$/.exec(line);
    if (match) found.push({ level: match[1].length as 2 | 3, text: plain(match[2]) });
  }
  return found;
}

let headingCache: Promise<Map<string, Heading[]>> | null = null;

/** Every chapter's headings. The bodies load once, on the first search. */
function loadHeadings(chapters: ChapterMeta[]): Promise<Map<string, Heading[]>> {
  headingCache ??= Promise.all(
    chapters.map((chapter) =>
      api
        .chapter(chapter.slug)
        .then((full) => [chapter.slug, headings(full.body)] as const)
        .catch(() => [chapter.slug, [] as Heading[]] as const),
    ),
  ).then((pairs) => new Map(pairs));
  return headingCache;
}

/**
 * The search dialog's entries: every chapter, and its section headings once
 * the dialog has opened and the chapter bodies have loaded.
 */
export function useSearchEntries(chapters: ChapterMeta[], open: boolean): SearchEntry[] {
  const [sections, setSections] = useState<Map<string, Heading[]>>(new Map());

  useEffect(() => {
    if (open && chapters.length) loadHeadings(chapters).then(setSections);
  }, [open, chapters]);

  return useMemo(() => {
    const numbers = chapterNumbers(chapters);
    return chapters.map((chapter) => {
      const href = `/c/${chapter.slug}`;
      // A `###` heading links to the `##` section above it.
      let section = 0;
      const found = (sections.get(chapter.slug) ?? []).map((heading) => {
        if (heading.level === 2) section += 1;
        return { text: heading.text, href: section ? `${href}#s-${section}` : href };
      });
      return {
        id: chapter.slug,
        href,
        title: chapter.title,
        kicker: `Chapter ${numbers.get(chapter.slug) ?? ""} · ${splitPart(chapter.part).name}`,
        summary: chapter.summary,
        keywords: [...chapter.objectives, chapter.part, chapter.slug],
        headings: found,
      };
    });
  }, [chapters, sections]);
}

/** The chapter to open next: the first one not yet done. */
export function nextChapter(
  chapters: ChapterMeta[],
  progress: Record<string, string>,
): ChapterMeta | undefined {
  return chapters.find((chapter) => progress[chapter.slug] !== "done");
}

export function doneCount(chapters: ChapterMeta[], progress: Record<string, string>): number {
  return chapters.filter((chapter) => progress[chapter.slug] === "done").length;
}

/** "Part 2 — A forward pass" → "02", or the 1-based index when unnumbered. */
export function partNumber(part: string, index: number): string {
  const label = splitPart(part).label;
  const digits = /\d+/.exec(label)?.[0];
  return (digits ?? String(index + 1)).padStart(2, "0");
}

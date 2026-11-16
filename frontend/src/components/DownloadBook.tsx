import { useEffect, useState } from "react";

import { api, type BookFile } from "../lib/api";

/**
 * The whole course as one PDF book: every chapter, every lab with its starter
 * file and hints, and the solutions in an appendix.
 *
 * The PDF is built at deploy time, so a server without one reports none, and
 * the sidebar shows no download link.
 */
export function useBook(enabled: boolean): BookFile | null {
  const [book, setBook] = useState<BookFile | null>(null);

  useEffect(() => {
    if (!enabled) return;
    api
      .book()
      .then((body) => setBook(body.book))
      .catch(() => undefined);
  }, [enabled]);

  return book;
}

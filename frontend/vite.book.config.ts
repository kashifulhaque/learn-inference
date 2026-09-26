import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The print layout for the PDF book, built apart from the site so it
// never ships in `dist`. scripts/build-book.mjs serves `dist-book` and prints it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "dist-book",
    emptyOutDir: true,
    rollupOptions: { input: "book.html" },
    // One page load, printed locally: bundle size doesn't matter here.
    chunkSizeWarningLimit: 4000,
  },
});

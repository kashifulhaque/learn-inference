// Every chapter figure, by the name a ```viz fence uses. Each group of chapters
// keeps its own index, so figures for different chapters never share a file.
// A name starts with its chapter's number, for example `10-roofline-explorer`;
// scripts/check_chapters.py reads the keys to catch a fence with a typo.

import type { ComponentType } from "react";
import architecture from "./architecture";
import foundations from "./foundations";
import kernels from "./kernels";
import performance from "./performance";
import scaling from "./scaling";

export type VizRegistry = Record<string, ComponentType>;

export const VIZ: VizRegistry = {
  ...foundations,
  ...architecture,
  ...performance,
  ...kernels,
  ...scaling,
};

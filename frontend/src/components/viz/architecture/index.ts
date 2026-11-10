import type { VizRegistry } from "../registry";
import DeltaOverwrite from "./delta-overwrite";
import ForgetHorizon from "./forget-horizon";
import GqaHeads from "./gqa-heads";
import KvEconomy from "./07a-kv-economy";
import LayerSchedule from "./layer-schedule";
import RotaryDials from "./rotary-dials";

// Figures for this group's chapters, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "05-rotary-dials": RotaryDials,
  "06-delta-overwrite": DeltaOverwrite,
  "06-forget-horizon": ForgetHorizon,
  "07-gqa-heads": GqaHeads,
  "07a-kv-economy": KvEconomy,
  "08-layer-schedule": LayerSchedule,
};

export default figures;

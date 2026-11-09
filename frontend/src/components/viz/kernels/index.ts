import type { VizRegistry } from "../registry";
import WarpCoalescing from "./12-warp-coalescing";
import TailGuard from "./12-tail-guard";
import FusionVsL2 from "./13-fusion-vs-l2";
import OnlineSoftmax from "./14-online-softmax";
import CausalBlocks from "./14-causal-blocks";
import SplitCount from "./14a-split-count";
import BlockTable from "./15-block-table";
import CheckpointTradeoff from "./15a-checkpoint-tradeoff";

// Figures for this group's chapters, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "12-tail-guard": TailGuard,
  "12-warp-coalescing": WarpCoalescing,
  "13-fusion-vs-l2": FusionVsL2,
  "14-online-softmax": OnlineSoftmax,
  "14-causal-blocks": CausalBlocks,
  "14a-split-count": SplitCount,
  "15-block-table": BlockTable,
  "15a-checkpoint-tradeoff": CheckpointTradeoff,
};

export default figures;

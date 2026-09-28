import type { VizRegistry } from "../registry";
import BatchTimeline from "./16-batch-timeline";
import ChunkBudget from "./16-chunk-budget";
import TailVsMean from "./17-tail-vs-mean";
import GroupQuant from "./18-group-quant";
import ColumnRowSplit from "./19-column-row-split";
import RingAllReduce from "./19-ring-all-reduce";
import AcceptResidual from "./20-accept-residual";
import SpeedupCurve from "./20-speedup-curve";

// Figures for this group's chapters, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "16-batch-timeline": BatchTimeline,
  "16-chunk-budget": ChunkBudget,
  "17-tail-vs-mean": TailVsMean,
  "18-group-quant": GroupQuant,
  "19-column-row-split": ColumnRowSplit,
  "19-ring-all-reduce": RingAllReduce,
  "20-accept-residual": AcceptResidual,
  "20-speedup-curve": SpeedupCurve,
};

export default figures;

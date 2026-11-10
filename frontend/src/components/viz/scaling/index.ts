import type { VizRegistry } from "../registry";
import BatchTimeline from "./16-batch-timeline";
import ChunkBudget from "./16-chunk-budget";
import ScheduleOverlap from "./16a-schedule-overlap";
import TailVsMean from "./17-tail-vs-mean";
import ThroughputLatency from "./17a-throughput-latency";
import GroupQuant from "./18-group-quant";
import GptqMargin from "./18a-gptq-margin";
import ColumnRowSplit from "./19-column-row-split";
import RingAllReduce from "./19-ring-all-reduce";
import AcceptResidual from "./20-accept-residual";
import SpeedupCurve from "./20-speedup-curve";

// Figures for this group's chapters, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "16-batch-timeline": BatchTimeline,
  "16-chunk-budget": ChunkBudget,
  "16a-schedule-overlap": ScheduleOverlap,
  "17-tail-vs-mean": TailVsMean,
  "17a-throughput-latency": ThroughputLatency,
  "18-group-quant": GroupQuant,
  "18a-gptq-margin": GptqMargin,
  "19-column-row-split": ColumnRowSplit,
  "19-ring-all-reduce": RingAllReduce,
  "20-accept-residual": AcceptResidual,
  "20-speedup-curve": SpeedupCurve,
};

export default figures;

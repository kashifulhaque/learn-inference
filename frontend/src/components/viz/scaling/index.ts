import type { VizRegistry } from "../registry";
import BatchTimeline from "./16-batch-timeline";
import ChunkBudget from "./16-chunk-budget";
import ScheduleOverlap from "./16a-schedule-overlap";
import RoutingTradeoff from "./16b-routing-tradeoff";
import TailVsMean from "./17-tail-vs-mean";
import ThroughputLatency from "./17a-throughput-latency";
import ChunkVsDisagg from "./17b-chunk-vs-disagg";
import GroupQuant from "./18-group-quant";
import GptqMargin from "./18a-gptq-margin";
import WandaError from "./18b-wanda-error";
import AdapterPool from "./18c-adapter-pool";
import ColumnRowSplit from "./19-column-row-split";
import RingAllReduce from "./19-ring-all-reduce";
import PipelineGantt from "./19b-pipeline-gantt";
import RingWork from "./19b-ring-work";
import DistinctExperts from "./19a-distinct-experts";
import AcceptResidual from "./20-accept-residual";
import SpeedupCurve from "./20-speedup-curve";
import BestTree from "./20a-best-tree";

// Figures for this group's chapters, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "16-batch-timeline": BatchTimeline,
  "16-chunk-budget": ChunkBudget,
  "16a-schedule-overlap": ScheduleOverlap,
  "16b-routing-tradeoff": RoutingTradeoff,
  "17-tail-vs-mean": TailVsMean,
  "17a-throughput-latency": ThroughputLatency,
  "17b-chunk-vs-disagg": ChunkVsDisagg,
  "18-group-quant": GroupQuant,
  "18a-gptq-margin": GptqMargin,
  "18b-wanda-error": WandaError,
  "18c-adapter-pool": AdapterPool,
  "19-column-row-split": ColumnRowSplit,
  "19-ring-all-reduce": RingAllReduce,
  "19b-pipeline-gantt": PipelineGantt,
  "19b-ring-work": RingWork,
  "19a-distinct-experts": DistinctExperts,
  "20-accept-residual": AcceptResidual,
  "20-speedup-curve": SpeedupCurve,
  "20a-best-tree": BestTree,
};

export default figures;

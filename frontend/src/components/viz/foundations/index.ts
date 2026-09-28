import type { VizRegistry } from "../registry";
import IntensityRidge from "./00-intensity-ridge";
import FloatFormats from "./00a-float-formats";
import LayerTypes from "./01-layer-types";
import SafetensorsOffsets from "./01-safetensors-offsets";
import BatchBudget from "./02-batch-budget";
import Breakeven from "./02-breakeven";
import StreamingDetokenizer from "./03-streaming-detokenizer";
import Bf16Stall from "./04-bf16-stall";
import RmsNormScaleShift from "./04-rmsnorm-scale-shift";

// Figures for this group's chapters, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "00-intensity-ridge": IntensityRidge,
  "00a-float-formats": FloatFormats,
  "01-safetensors-offsets": SafetensorsOffsets,
  "01-layer-types": LayerTypes,
  "02-breakeven": Breakeven,
  "02-batch-budget": BatchBudget,
  "03-streaming-detokenizer": StreamingDetokenizer,
  "04-rmsnorm-scale-shift": RmsNormScaleShift,
  "04-bf16-stall": Bf16Stall,
};

export default figures;

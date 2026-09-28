import type { VizRegistry } from "../registry";
import CacheMemory from "./CacheMemory";
import CacheWorkGrid from "./CacheWorkGrid";
import RooflineExplorer from "./RooflineExplorer";
import TemperatureOrder from "./TemperatureOrder";
import TruncationRules from "./TruncationRules";
import WaveQuantization from "./WaveQuantization";

// Figures for this group's chapters, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "09-cache-work-grid": CacheWorkGrid,
  "09-cache-memory": CacheMemory,
  "10-roofline-explorer": RooflineExplorer,
  "10-wave-quantization": WaveQuantization,
  "11-truncation-rules": TruncationRules,
  "11-temperature-order": TemperatureOrder,
};

export default figures;

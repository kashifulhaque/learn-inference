import { useState } from "react";
import { Figure, Segmented, Stat, Legend, color, useWidth, wash } from "../kit";

// Constants from chapter 12, "Coalescing, sector by sector": a warp is 32
// lanes, each reads one 4-byte float32, and the memory system moves whole
// 32-byte sectors.
const LANES = 32;
const FLOAT_BYTES = 4;
const SECTOR_BYTES = 32;
const SLOTS = SECTOR_BYTES / FLOAT_BYTES; // 8 floats per sector
const USEFUL_BYTES = LANES * FLOAT_BYTES; // 128 bytes the warp asked for

// Strides in floats between neighboring lanes. 1 is `a[i]`, 32 is the lab's
// `a[i * 32]`, and 4 matches the per-thread chunk of four columns the chapter's
// RMSNorm section warns about (16 sectors for 128 useful bytes).
const STRIDES = [1, 2, 4, 8, 32] as const;

type Sector = { id: number; used: Set<number>; firstLane: number; lastLane: number };

/** The sectors a warp touches when lane t reads float element t * stride. */
function sectorsFor(stride: number): Sector[] {
  const byId = new Map<number, Sector>();
  for (let lane = 0; lane < LANES; lane += 1) {
    const byte = lane * stride * FLOAT_BYTES;
    const id = Math.floor(byte / SECTOR_BYTES);
    const slot = (byte % SECTOR_BYTES) / FLOAT_BYTES;
    const sector = byId.get(id) ?? { id, used: new Set<number>(), firstLane: lane, lastLane: lane };
    sector.used.add(slot);
    sector.lastLane = lane;
    byId.set(id, sector);
  }
  return [...byId.values()];
}

// Eight sectors per row, four rows: room for the worst case, one sector per lane.
const COLUMNS = 8;
const HEIGHT = 216;

export default function WarpCoalescing() {
  const [stride, setStride] = useState<number>(32);
  const [ref, width] = useWidth();

  const sectors = sectorsFor(stride);
  const fetched = sectors.length * SECTOR_BYTES;
  const efficiency = USEFUL_BYTES / fetched;

  // Lanes: one row of 32 cells across the full width.
  const laneW = width / LANES;
  const laneY = 18;
  const laneH = 14;
  const bracketY = laneY + laneH + 5;

  // Sectors: a grid of 8 columns, each sector drawn as 8 float slots.
  const gap = 6;
  const gridY = 84;
  const crateW = (width - gap * (COLUMNS - 1)) / COLUMNS;
  const crateH = 22;
  const rowGap = 8;
  const slotW = crateW / SLOTS;

  return (
    <Figure
      title="How many sectors one warp's load fetches"
      controls={
        <Segmented
          label="Stride between lanes"
          value={stride}
          options={STRIDES.map((s) => ({ value: s, label: s === 1 ? "1 float" : `${s} floats` }))}
          onChange={setStride}
        />
      }
      readout={
        <>
          <Stat label="Stride" value={`${stride} ${stride === 1 ? "float" : "floats"}, ${stride * FLOAT_BYTES} B`} />
          <Stat label="Sectors fetched" value={`${sectors.length} of 32 max`} tone={color.b} />
          <Stat label="Bytes fetched" value={`${fetched} B for 128 B`} />
          <Stat label="Efficiency" value={`${(efficiency * 100).toFixed(1)}%`} tone={efficiency < 1 ? color.bad : color.ok} />
          <Stat label="Read amplification" value={`${fetched / USEFUL_BYTES}x`} />
        </>
      }
      caption={
        <>
          Each box is one 32-byte sector, and each lane's float fills one eighth of it. From a stride of
          8 floats up, every lane lands in its own sector, so the warp fetches 1024 bytes for 128: the 8x
          read amplification of <code>a[i * 32]</code>.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`A warp of 32 lanes reads floats ${stride} apart and touches ${sectors.length} sectors, fetching ${fetched} bytes to deliver 128, ${(efficiency * 100).toFixed(1)} percent efficiency.`}
        >
          <text x={0} y={12}>
            32 lanes of one warp, each reading one float
          </text>
          {Array.from({ length: LANES }, (_, lane) => (
            <rect
              key={lane}
              x={lane * laneW + 0.5}
              y={laneY}
              width={Math.max(1, laneW - 1)}
              height={laneH}
              rx={1.5}
              style={{ fill: wash(color.b, 35), stroke: color.b, strokeWidth: 0.75 }}
            />
          ))}
          {/* Brackets group the lanes that share a sector. */}
          {sectors.map((sector) => {
            const x0 = sector.firstLane * laneW + 1;
            const x1 = (sector.lastLane + 1) * laneW - 1;
            return (
              <path
                key={sector.id}
                d={`M${x0},${bracketY} v4 H${x1} v-4`}
                style={{ fill: "none", stroke: color.muted, strokeWidth: 1 }}
              />
            );
          })}
          <text x={0} y={bracketY + 18}>
            lane 0
          </text>
          <text x={width} y={bracketY + 18} textAnchor="end">
            lane 31
          </text>
          <text x={width / 2} y={bracketY + 18} textAnchor="middle">
            {sectors.length === LANES ? "one sector per lane" : `${LANES / sectors.length} lanes per sector`}
          </text>

          <text x={0} y={gridY - 8}>
            Sectors fetched: {sectors.length}
          </text>
          {Array.from({ length: LANES }, (_, index) => {
            const col = index % COLUMNS;
            const row = Math.floor(index / COLUMNS);
            const x = col * (crateW + gap);
            const y = gridY + row * (crateH + rowGap);
            const sector = sectors[index];
            if (!sector) {
              return (
                <rect
                  key={index}
                  x={x + 0.5}
                  y={y + 0.5}
                  width={crateW - 1}
                  height={crateH - 1}
                  rx={2}
                  style={{ fill: "none", stroke: color.line, strokeDasharray: "2 3" }}
                />
              );
            }
            return (
              <g key={index}>
                {Array.from({ length: SLOTS }, (_, slot) => (
                  <rect
                    key={slot}
                    x={x + slot * slotW}
                    y={y}
                    width={slotW}
                    height={crateH}
                    style={{
                      fill: sector.used.has(slot) ? color.b : wash(color.bad, 12),
                      stroke: "none",
                    }}
                  />
                ))}
                <rect
                  x={x + 0.5}
                  y={y + 0.5}
                  width={crateW - 1}
                  height={crateH - 1}
                  rx={2}
                  style={{ fill: "none", stroke: color.lineStrong }}
                />
              </g>
            );
          })}
        </svg>
        <Legend
          items={[
            { label: "Float a lane asked for", tone: color.b },
            { label: "Fetched and thrown away", tone: wash(color.bad, 30) },
            { label: "Not fetched", tone: color.line },
          ]}
        />
      </div>
    </Figure>
  );
}

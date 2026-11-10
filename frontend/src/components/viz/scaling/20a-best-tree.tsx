import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth, wash } from "../kit";

// Best tree for a node budget, from scripts/figures/20a-tree-speculation.py.
// The acceptance table is closed-form for a Zipf target and a noisy draft:
// ACCEPT[d - 1][k] is the chance the k-th child at depth d is the accepted one.
// These are computed, not measured, acceptance rates.
const ACCEPT = [
  [0.795, 0.048, 0.028, 0.022],
  [0.658, 0.08, 0.032, 0.028],
  [0.628, 0.088, 0.056, 0.037],
  [0.395, 0.149, 0.089, 0.06],
  [0.463, 0.078, 0.065, 0.053],
  [0.531, 0.085, 0.058, 0.031],
] as const;

const TREES = [
  { budget: 1, tree: 0.795, chain: 0.795, parents: [-1, 0] },
  { budget: 2, tree: 1.318, chain: 1.318, parents: [-1, 0, 1] },
  { budget: 4, tree: 1.775, chain: 1.775, parents: [-1, 0, 1, 2, 3] },
  { budget: 6, tree: 1.899, chain: 1.867, parents: [-1, 0, 1, 2, 3, 1, 4] },
  { budget: 8, tree: 1.996, chain: 1.867, parents: [-1, 0, 1, 2, 3, 1, 4, 3, 0] },
  { budget: 12, tree: 2.146, chain: 1.867, parents: [-1, 0, 1, 2, 3, 1, 4, 3, 0, 2, 5, 6, 8] },
  {
    budget: 16,
    tree: 2.258,
    chain: 1.867,
    parents: [-1, 0, 1, 2, 3, 1, 4, 3, 0, 2, 5, 6, 8, 3, 2, 0, 1],
  },
] as const;

const BUDGETS = TREES.map((item) => item.budget);

function layout(parents: readonly number[]) {
  const n = parents.length;
  const depth: number[] = [];
  const rank: number[] = [];
  const seen: Record<number, number> = {};
  parents.forEach((parent) => {
    depth.push(parent < 0 ? 0 : depth[parent] + 1);
    rank.push(parent < 0 ? 0 : (seen[parent] = (seen[parent] ?? -1) + 1));
  });
  const prob: number[] = [1];
  for (let i = 1; i < n; i += 1) prob.push(prob[parents[i]] * ACCEPT[depth[i] - 1][rank[i]]);
  const children: number[][] = parents.map(() => []);
  parents.forEach((parent, i) => {
    if (parent >= 0) children[parent].push(i);
  });
  const row: number[] = new Array(n).fill(0);
  let next = 0;
  const walk = (node: number) => {
    if (children[node].length === 0) {
      row[node] = next;
      next += 1;
      return;
    }
    children[node].forEach(walk);
    row[node] = (row[children[node][0]] + row[children[node][children[node].length - 1]]) / 2;
  };
  walk(0);
  return { depth, prob, row, rows: next, maxDepth: Math.max(...depth) };
}

export default function BestTree() {
  const [budget, setBudget] = useState(8);
  const [ref, width] = useWidth();
  const item = TREES.find((entry) => entry.budget === budget) ?? TREES[4];
  const { depth, prob, row, rows, maxDepth } = layout(item.parents);
  const left = 24;
  const right = 24;
  const top = 18;
  const gap = 30;
  const height = top * 2 + Math.max(rows - 1, 1) * gap;
  const innerW = Math.max(width - left - right, 80);
  const xOf = (d: number) => left + (maxDepth === 0 ? 0 : (d / maxDepth) * innerW);
  const yOf = (r: number) => top + r * gap;

  return (
    <Figure
      title="The best tree for a node budget"
      controls={
        <Slider
          label="Draft nodes"
          values={BUDGETS}
          value={budget}
          onChange={setBudget}
          format={(value) => `${value} nodes`}
        />
      }
      readout={
        <>
          <Stat label="Tree, accepted" value={item.tree.toFixed(3)} tone={color.b} />
          <Stat label="Chain, accepted" value={item.chain.toFixed(3)} />
          <Stat label="Depth" value={`${maxDepth} levels`} />
        </>
      }
      caption="Each node shows the chance that it is on the accepted path. Larger nodes are likelier. The greedy pick adds the likeliest next node, so the tree grows sideways once a deeper edge is less likely than a sibling. The chain stops at the table's six levels. Computed values, not measured acceptance."
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`Best draft tree with ${budget} nodes and expected accepted length ${item.tree.toFixed(3)}`}
        >
          {item.parents.map((parent, i) =>
            parent < 0 ? null : (
              <line
                key={`edge-${i}`}
                x1={xOf(depth[parent])}
                y1={yOf(row[parent])}
                x2={xOf(depth[i])}
                y2={yOf(row[i])}
                stroke={color.lineStrong}
                strokeWidth={1.2}
              />
            ),
          )}
          {item.parents.map((_, i) => (
            <g key={`node-${i}`}>
              <circle
                cx={xOf(depth[i])}
                cy={yOf(row[i])}
                r={5 + 7 * prob[i]}
                fill={wash(color.b, 18 + Math.round(60 * prob[i]))}
                stroke={color.b}
                strokeWidth={1.2}
              />
              <text
                x={xOf(depth[i])}
                y={yOf(row[i]) - 14}
                textAnchor="middle"
                fill={color.muted}
                fontSize={10}
              >
                {i === 0 ? "root" : prob[i].toFixed(2)}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </Figure>
  );
}

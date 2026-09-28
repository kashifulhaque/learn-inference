import { useState } from "react";
import { Figure, Segmented, Stat, color, useWidth, wash } from "../kit";

// Chapter 3, "Detokenization is stateful". Like the lab's harness, this
// tokenizer chops the UTF-8 bytes into fixed two-byte runs, which splits क
// (E0 A4 95) as the chapter describes: E0 A4 in one token, 95 in the next.
const TOKEN_BYTES = 2;
const TEXTS = {
  devanagari: { label: "hi, क!", text: "hi, क!" },
  emoji: { label: "ok 👍", text: "ok 👍" },
} as const;
type TextName = keyof typeof TEXTS;

const REPLACEMENT = "�";

/** Decodes bytes as UTF-8, replacing invalid or incomplete sequences with U+FFFD. */
const decodeReplace = (bytes: Uint8Array) => new TextDecoder("utf-8").decode(bytes);

/** The lab's push(): decode every id so far, drop an incomplete tail, emit the new suffix. */
function streamWithState(tokens: Uint8Array[]): string[] {
  const out: string[] = [];
  let emitted = 0;
  const seen: number[] = [];
  for (const token of tokens) {
    seen.push(...token);
    // errors="ignore": drop what can't be decoded yet. This text has no real U+FFFD to lose.
    const text = decodeReplace(Uint8Array.from(seen)).split(REPLACEMENT).join("");
    out.push(text.slice(emitted));
    emitted = text.length;
  }
  return out;
}

const hex = (bytes: Uint8Array) =>
  Array.from(bytes, (b) => b.toString(16).toUpperCase().padStart(2, "0")).join(" ");
const visible = (s: string) => s.replace(/ /g, "␣");

const LABEL_W = 76;
const ROW_H = 30;
const HEAD_H = 18;

export default function StreamingDetokenizer() {
  const [ref, width] = useWidth();
  const [name, setName] = useState<TextName>("devanagari");
  const { text } = TEXTS[name];

  const bytes = new TextEncoder().encode(text);
  const tokens: Uint8Array[] = [];
  for (let i = 0; i < bytes.length; i += TOKEN_BYTES) tokens.push(bytes.slice(i, i + TOKEN_BYTES));
  const alone = tokens.map(decodeReplace);
  const stateful = streamWithState(tokens);
  const empties = stateful.filter((s) => s === "").length;

  const colW = (width - LABEL_W) / tokens.length;
  const rows = [
    { label: "Token bytes", cells: tokens.map(hex), mono: true },
    { label: "Each alone", cells: alone.map(visible), mono: false },
    { label: "With state", cells: stateful.map((s) => (s === "" ? "none" : visible(s))), mono: false },
  ];
  const height = HEAD_H + rows.length * ROW_H + 2;

  return (
    <Figure
      title="Streaming text back, token by token"
      controls={
        <Segmented
          label="Text"
          value={name}
          onChange={setName}
          options={Object.entries(TEXTS).map(([value, t]) => ({ value: value as TextName, label: t.label }))}
        />
      }
      readout={
        <>
          <Stat label="Text" value={text} />
          <Stat label="UTF-8 bytes" value={`${bytes.length}, in ${tokens.length} two-byte tokens`} />
          <Stat label="Screen, each token alone" value={alone.join("")} tone={color.bad} />
          <Stat label="Screen, with state" value={stateful.join("")} tone={color.ok} />
          <Stat label="Pushes that emit nothing" value={String(empties)} />
        </>
      }
      caption={
        <>
          Two-byte tokens split multibyte characters across pushes. Decoding each token alone prints replacement
          characters at the seam; decoding the accumulated ids emits nothing until the character is complete, then
          emits all of it at once.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`The text ${text} as ${tokens.length} two-byte tokens. Decoded one token at a time it reads ${alone.join("")}; decoded with state it reads ${stateful.join("")}.`}
        >
          {tokens.map((_, k) => (
            <text key={k} x={LABEL_W + (k + 0.5) * colW} y={12} textAnchor="middle" style={{ fill: color.subtle }}>
              {`push ${k + 1}`}
            </text>
          ))}
          {rows.map((row, r) => {
            const y = HEAD_H + r * ROW_H;
            return (
              <g key={row.label}>
                <text x={0} y={y + 19} style={{ fill: color.muted }}>
                  {row.label}
                </text>
                {row.cells.map((cell, k) => {
                  const bad = r === 1 && cell.includes(REPLACEMENT);
                  const empty = r === 2 && stateful[k] === "";
                  const tone = bad ? color.bad : r === 2 ? color.ok : color.lineStrong;
                  return (
                    <g key={k}>
                      <rect
                        x={LABEL_W + k * colW + 2}
                        y={y + 2}
                        width={colW - 4}
                        height={ROW_H - 6}
                        rx={3}
                        strokeDasharray={empty ? "3 2" : undefined}
                        style={{
                          fill: bad ? wash(color.bad, 12) : r === 2 && !empty ? wash(color.ok, 10) : color.well,
                          stroke: tone,
                        }}
                      />
                      <text
                        x={LABEL_W + (k + 0.5) * colW}
                        y={y + 19}
                        textAnchor="middle"
                        style={{
                          fill: empty ? color.faint : color.fg,
                          fontFamily: row.mono ? "var(--font-mono)" : undefined,
                        }}
                      >
                        {cell}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
    </Figure>
  );
}

import { useState } from "react";
import { Figure, Segmented, Stat, color, wash } from "../kit";

// Data printed by `python3 scripts/figures/11a-constrained-decoding.py --viz`,
// which runs the lab solution. The pattern is (yes|no). The prefix is the text
// generated so far, and `tokens` says which of ten sample tokens the mask allows.
// A prefix is empty at the start. This is a CPU computation, not a GPU one.
const DATA = {
  "pattern": "(yes|no)",
  "vocab": 296,
  "rows": [
    {
      "prefix": "",
      "legal": 5,
      "tokens": {
        "y": true,
        "n": true,
        "ye": true,
        "es": false,
        "yes": true,
        "no": true,
        "maybe": false,
        "e": false,
        "o": false,
        "s": false,
        "END": false
      }
    },
    {
      "prefix": "y",
      "legal": 2,
      "tokens": {
        "y": false,
        "n": false,
        "ye": false,
        "es": true,
        "yes": false,
        "no": false,
        "maybe": false,
        "e": true,
        "o": false,
        "s": false,
        "END": false
      }
    },
    {
      "prefix": "ye",
      "legal": 1,
      "tokens": {
        "y": false,
        "n": false,
        "ye": false,
        "es": false,
        "yes": false,
        "no": false,
        "maybe": false,
        "e": false,
        "o": false,
        "s": true,
        "END": false
      }
    },
    {
      "prefix": "yes",
      "legal": 1,
      "tokens": {
        "y": false,
        "n": false,
        "ye": false,
        "es": false,
        "yes": false,
        "no": false,
        "maybe": false,
        "e": false,
        "o": false,
        "s": false,
        "END": true
      }
    },
    {
      "prefix": "n",
      "legal": 1,
      "tokens": {
        "y": false,
        "n": false,
        "ye": false,
        "es": false,
        "yes": false,
        "no": false,
        "maybe": false,
        "e": false,
        "o": true,
        "s": false,
        "END": false
      }
    },
    {
      "prefix": "no",
      "legal": 1,
      "tokens": {
        "y": false,
        "n": false,
        "ye": false,
        "es": false,
        "yes": false,
        "no": false,
        "maybe": false,
        "e": false,
        "o": false,
        "s": false,
        "END": true
      }
    }
  ]
} as const;

const ORDER = ["y", "n", "ye", "es", "yes", "no", "maybe", "e", "o", "s", "END"] as const;

export default function TokenMask() {
  const [index, setIndex] = useState(0);
  const row = DATA.rows[index];
  const tokens = row.tokens as Record<string, boolean>;
  const label = (prefix: string) => (prefix === "" ? "start" : `"${prefix}"`);

  return (
    <Figure
      title="One DFA state, one token mask"
      controls={
        <Segmented
          label="Text so far"
          value={index}
          options={DATA.rows.map((r, i) => ({ value: i, label: label(r.prefix) }))}
          onChange={setIndex}
        />
      }
      readout={
        <>
          <Stat label="Legal tokens" value={`${row.legal} of ${DATA.vocab}`} tone={color.ok} />
          <Stat label="Pattern" value={DATA.pattern} />
        </>
      }
      caption="Each cell is a token from the toy vocabulary. A token is legal when all of its bytes keep the pattern alive. END is legal only when the text so far already matches. Values come from the lab solution."
    >
      <div
        role="list"
        aria-label="Sample tokens and whether the mask allows them"
        style={{ display: "flex", flexWrap: "wrap", gap: 8 }}
      >
        {ORDER.map((name) => {
          const legal = tokens[name];
          return (
            <div
              key={name}
              role="listitem"
              style={{
                padding: "6px 10px",
                borderRadius: 6,
                border: `1px solid ${legal ? color.ok : color.line}`,
                background: legal ? wash(color.ok, 22) : "transparent",
                color: legal ? color.fg : color.faint,
                textDecoration: legal ? "none" : "line-through",
                fontFamily: "var(--mono, monospace)",
                fontSize: 13,
              }}
            >
              {name}
              <span style={{ marginLeft: 6, fontFamily: "inherit" }}>{legal ? "legal" : "masked"}</span>
            </div>
          );
        })}
      </div>
    </Figure>
  );
}

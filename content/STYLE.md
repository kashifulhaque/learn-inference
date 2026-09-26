# How to write a chapter

A chapter teaches one idea well enough that the reader can build it in the lab.
The reader is a capable engineer reading on a screen, often beside the lab
editor. They skim first, then read closely. Write for both passes: someone who
reads only the callouts and the highlighted phrases leaves with the argument,
and someone who reads every line gets the full derivation.

The prose follows the
[Google developer documentation style guide](https://developers.google.com/style).
This file covers what that guide doesn't: the reading aids the app renders, and
how a chapter is shaped.

## The shape of a chapter

Every chapter follows the same outline, so a reader always knows where they are:

1. The `#` title. The page header shows it, and the body hides this copy, but
   keep it so the file reads on its own.
2. A `[!TLDR]` callout of three to five bullets: the whole chapter in one
   minute.
3. A short opening, two or three paragraphs at most, that says why the topic
   matters for the engine.
4. `## Before you start`, if the chapter needs background. Give each refresher a
   bold run-in term and one to three sentences. Move anything longer into a
   `[!DEEPDIVE]`.
5. The teaching sections, each a `##` heading. The app numbers them and tracks
   which one the reader is in, so each one should answer one question.
6. `## What goes wrong`, for the mistakes the lab is likely to provoke.
7. A `[!RECAP]` callout, as the last thing before the questions.
8. `## Check your understanding`, one `[!QUESTION]` per question.
9. `## Lab`, which opens with a `[!TRY]` callout.
10. `## Further reading`.

## Write for a screen

- Keep paragraphs to four sentences or fewer. One idea per paragraph.
- Open each `##` section with one sentence that says what the section answers.
- Put a number, a comparison, or a sequence in a list or a table instead of a
  paragraph. A table needs three or more columns to earn its place.
- Lead with the conclusion, then justify it. "RMSNorm is bandwidth bound: it
  does 1 FLOP per byte against a ridge point of 161" beats building up to it.
- Name the symbol and the thing together the first time: "the gain $w$," not
  "$w$."
- Use contractions, the second person, the present tense, and the active voice.

## Maths

Show the maths where it carries the argument, and walk through it one step at a
time. The pattern for every display equation is: say in words what you are
about to write, write it, then say what it means.

- **Before the equation**, one sentence of intent: "Add up the squares, then
  divide by the width:"
- **After the equation**, a plain-language reading, either one sentence or, for a
  dense result, an `[!INTUITION]` callout.
- **Box the result** a derivation arrives at with `\boxed{}`, so the eye lands on
  it.
- **Colour the parts** of an equation that has several named pieces, and use the
  same colour when the prose refers to them. Four macros exist:
  `\hla{}` (green), `\hlb{}` (blue), `\hlc{}` (amber), and `\hld{}` (rose). For
  example, write `$\hla{\epsilon}$` in the prose that explains the
  $\epsilon$ in `$$y_i = \frac{x_i}{\sqrt{\operatorname{ms}(x) + \hla{\epsilon}}}$$`.
  Keep the colour of a symbol the same across the chapter, and colour at most
  three things in one equation.
- **Plug in real numbers** in an `[!EXAMPLE]` callout once a formula is general:
  the reader believes $d = 5120$ more than $d$.
- **Keep long derivations**, but when a chain runs past about four display
  equations, keep the setup and the result in the body and put the steps in a
  `[!DEEPDIVE]`. The conclusion must never live only inside a deep dive.

The site renders maths with KaTeX, which supports a subset of LaTeX. Check every
expression before you commit:

```bash
node scripts/check_math.mjs content/chapters/CHAPTER.md
```

## Reading aids

### Callouts

A callout is a blockquote whose first line is a marker. Text after the marker, on
the same line, is the callout's title. Every following line keeps the `>`
prefix, including blank lines, display maths, and code blocks:

```markdown
> [!KEY] Addends vanish at a ratio of 256
> Once the running sum is 256 times the next term, bfloat16 rounds the
> addition away entirely:
>
> $$
> \frac{S}{a} \ge 2^{8} \implies S + a = S
> $$
```

| Marker | Renders as | Use it for |
|---|---|---|
| `[!TLDR]` | In one minute | The chapter summary, straight after the title. One per chapter. |
| `[!KEY]` | Key idea, highlighted | The one thing to carry out of a section. At most one per section. |
| `[!INTUITION]` | In plain words | A plain-language reading of dense maths. |
| `[!EXAMPLE]` | Worked example | A formula with the model's real numbers plugged in. |
| `[!NOTE]` | Note | An aside that helps but that the argument doesn't need. |
| `[!TIP]` | Tip | Practical advice for the lab or for debugging. |
| `[!WARNING]` | Watch out | A mistake with a silent or expensive failure. |
| `[!DEEPDIVE]` | Deep dive, collapsed | A full derivation or a tangent. Always give it a title. |
| `[!QUESTION]` | Check yourself, collapsed | A question as the title, with the answer as the body. |
| `[!RECAP]` | What to remember | Four to six bullets before the questions. One per chapter. |
| `[!TRY]` | Try it | What the lab asks you to build, and what passing looks like. |

A callout draws the eye, so a page full of them draws it nowhere. Aim for no more
than one callout every screen or so of prose, don't put two of the same kind
next to each other, except `[!QUESTION]`, and never put the only statement of a
fact inside a collapsed callout.

To check a chapter's callouts and outline, run the following command:

```bash
python3 scripts/check_chapters.py content/chapters/CHAPTER.md
```

### Highlights

Wrap the single phrase in a section that a skimming reader must not miss in
`==double equals signs==`. It renders as a highlighter mark. Use at most two per
section, and highlight a claim, not a whole sentence:

```markdown
The error is ==a bias, not noise==, so averaging over 5120 terms doesn't
cancel it.
```

### Everything else

- Code blocks get a language label and a copy button. Tag every fence with its
  language.
- Link to another chapter with its route, for example
  `[the roofline chapter](/c/10-roofline)`, so it opens in place.
- A `*term*` in italics marks a term on first use. Don't italicize for emphasis;
  use a highlight or restructure the sentence instead.

## Accuracy

Every number in a chapter is either measured on the target A100, arithmetic from
the model's config, or cited. When you revise a chapter, keep every number,
measurement, code sample, lab requirement, and link intact, and recompute any
new arithmetic you add. If a measurement contradicts the tidy prediction, say
so: the chapters exist partly to show where the textbook byte count is wrong.

# Terminal chat samples

These are synthetic presentation fixtures, not answers verified against a real BOQ.
The live model chooses wording; the prompt requests bold values and compact Markdown tables.
The renderer preserves values and never computes or extracts amounts from prose.
In a color terminal, headings use subtle blue, values are bold, and citations are dimmed.

## Quantity query

Question: What is the quantity of RC raft foundation concrete C35/45?

```text
820.75 m³ of RC raft foundation concrete C35/45.

Sources
Example.xlsx
  sheet Concrete, row 8
```

## Financial query

Question: What is the recorded concrete subtotal?

```text
USD 154,956.70 — recorded concrete subtotal.

Sources
Example.xlsx
  sheet Concrete, row 11

Notes
• This is the concrete subtotal, not the full project total.
```

## Multi-item comparison

Question: Compare the raft and slab quantities and amounts.

```text
Item   Quantity  Amount (USD)
─────────────────────────────
Raft  820.75 m³    123,456.70
Slab  210.00 m³     31,500.00

Sources
Example.xlsx
  sheet Concrete, rows 8–9
```

## Comparison at 32 columns

The same parsed table cells switch to labelled values. No values are omitted.

```text
Item: Raft
Quantity: 820.75 m³
Amount (USD): 123,456.70

Item: Slab
Quantity: 210.00 m³
Amount (USD): 31,500.00

Sources
Example.xlsx
  sheet Concrete, rows 8–9
```

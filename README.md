# Document corpus

Put the PDFs (or .txt / .md files) you want the assistant to answer from in this folder.

Use only **public** documents. Do not put anything from client engagements here.

Good public sources to collect yourself (search for the document titles, and download
the current versions from the publisher's own site):

- NatWest personal banking: fees and charges / "Fees and information" leaflets, account
  terms and conditions, current account and savings product guides, help-centre pages
  saved as PDF, the Consumer Duty or complaints-handling pages.
- FCA: Consumer Duty guidance, complaints-handling rules (DISP) summaries.
- Bank of England / UK Finance / Financial Ombudsman Service: consumer guides.
- Anything on scams and fraud prevention that the bank publishes for customers.

## Naming convention (sets the `doc_type` metadata used in filtering and citations)

| Prefix        | doc_type     |
|---------------|--------------|
| `fees_`       | fees         |
| `terms_`      | terms        |
| `regulatory_` | regulatory   |
| `faq_`        | faq          |
| `product_`    | product      |
| anything else | general      |

Example: `fees_current-account-fees-and-information.pdf`.

Keep a note of the date you downloaded each file. Fees and terms change, and the
assistant tells users to check current terms with the bank, but your README should
say which version the demo was built on.

Scanned PDFs (images with no text layer) are skipped with a warning. They need OCR first.

Then run `python -m app.ingest --reset` (or click "Rebuild index" in the app).

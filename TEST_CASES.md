# Manual Test Cases

Use these cases before the demonstration. Exact scores depend on the knowledge-base text and scikit-learn version.

| # | Input | Expected behavior |
|---|---|---|
| 1 | The payment API is returning a 500 error | Retrieve KB001, show diagnostic checks and Backend Engineering escalation |
| 2 | My application cannot connect to MySQL | Retrieve KB002 and show database checks |
| 3 | Login gives me an unauthorized error | Retrieve KB003 and/or KB005, show supporting entry |
| 4 | The API route returns Not Found | Retrieve KB004 |
| 5 | Our app is slow today | Retrieve KB006 |
| 6 | Service is unavailable and returns 503 | Retrieve KB007 |
| 7 | Remote API calls time out | Retrieve KB008 |
| 8 | The hostname cannot be resolved | Retrieve KB009 |
| 9 | File upload fails because it is too large | Retrieve KB010 |
| 10 | The app runs out of memory | Retrieve KB011 |
| 11 | API throttling returns 429 | Retrieve KB012 |
| 12 | Errors started immediately after release | Retrieve KB013 |
| 13 | The request payload is invalid | Retrieve KB014 |
| 14 | Queue jobs are stuck | Retrieve KB015 |
| 15 | How do I bake sourdough bread? | Ideally reject as insufficient knowledge; tune and test the retrieval threshold |

## Notes

- Check the displayed source, not just the generated answer.
- Verify that causes are presented as possibilities unless confirmed by evidence.
- Test with and without `GEMINI_API_KEY`.
- Test API failure or rate-limit handling if using Gemini.
- The current threshold in `app.py` is a starting heuristic. Evaluate false matches and adjust it before the demo.

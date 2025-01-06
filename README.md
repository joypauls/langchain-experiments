# PDF Assistant

Experimenting with LangChain. Project uses Poetry.

## Setup

Requires a `.env` file with the `OPENAI_API_KEY` variable set. Run `touch .env` at the project root and add the following line:
```
OPENAI_API_KEY=<your key>
```
Remember not to commit this to source control.

Add `USER_AGENT=` to silence a warning from LangChain

import { useEffect, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const MIN_QUESTION = 5
const MAX_QUESTION = 1000

const EXAMPLES = [
  'What share of finance jobs had a defined contribution pension in 2024?',
  'How do contribution rates for small employers compare to large ones?',
  'Has participation in construction gone up since 2019?',
]

const TOOL_LABELS = {
  list_tables: 'Listed available tables',
  describe_schema: 'Described a table',
  list_dimension_values: 'Listed dimension values',
  lookup_industry_membership_rate: 'Looked up an industry membership rate',
  industry_participation_trend: 'Compared an industry trend',
  compare_contribution_bands_by_size: 'Compared contribution bands by employer size',
  explain_quality_flags: 'Checked quality flags',
}

function DatasetPanel({ dataset }) {
  if (dataset.state === 'loading') {
    return <p className="muted" role="status">Checking the dataset…</p>
  }
  if (dataset.state === 'unreachable') {
    return (
      <p className="notice notice-error" role="alert">
        The API at <code>{API_BASE}</code> did not respond. Start the backend, then reload this page.
      </p>
    )
  }
  const info = dataset.info
  if (!info.ready) {
    return (
      <>
        <h2 id="dataset-heading">No dataset yet</h2>
        <p className="muted">Build a snapshot before asking a question. The README shows the one-line command.</p>
      </>
    )
  }
  const isFixture = info.mode === 'fixture'
  return (
    <>
      <h2 id="dataset-heading">{isFixture ? 'Synthetic test fixture' : 'Verified ONS snapshot'}</h2>
      <dl className="figures">
        <div><dt>observations</dt><dd>{info.rows.toLocaleString('en-GB')}</dd></div>
        <div><dt>tables</dt><dd>{info.tables.length}</dd></div>
        <div><dt>latest year</dt><dd>{info.years.at(-1)}</dd></div>
      </dl>
      <p className="small">
        {isFixture
          ? info.notice
          : 'All employee jobs. Coverage: 2009–2021 and 2024; availability varies by table. Contribution methodology changes across editions.'}
      </p>
      {info.agent_configured ? null : (
        <p className="notice" role="status">
          The data is ready, but no model provider is configured, so questions are switched off.
        </p>
      )}
    </>
  )
}

function ToolTrace({ calls }) {
  if (calls.length === 0) return null
  return (
    <details className="trace">
      <summary>How this answer was found ({calls.length} {calls.length === 1 ? 'lookup' : 'lookups'})</summary>
      <ol>
        {calls.map((call, index) => (
          <li key={index} className={call.ok ? 'ok' : 'failed'}>
            <span className="trace-name">{TOOL_LABELS[call.tool] || call.tool}</span>
            <span className="trace-status">{call.ok ? 'returned data' : 'no result'}</span>
            {call.sources?.length ? <span className="trace-sources">{call.sources.join(', ')}</span> : null}
          </li>
        ))}
      </ol>
    </details>
  )
}

function App() {
  const [dataset, setDataset] = useState({ state: 'loading' })
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    fetch(`${API_BASE}/ready`, { signal: controller.signal })
      .then((response) => response.json())
      .then((info) => setDataset({ state: 'loaded', info }))
      .catch(() => {
        if (!controller.signal.aborted) setDataset({ state: 'unreachable' })
      })
    return () => controller.abort()
  }, [])

  const serviceReady = dataset.state === 'loaded' && dataset.info.ready && dataset.info.agent_configured
  const trimmed = question.trim()
  const canSubmit = serviceReady && !loading && trimmed.length >= MIN_QUESTION

  const askAgent = async (event) => {
    event.preventDefault()
    if (!canSubmit) return
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed }),
      })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) {
        const detail = typeof payload.detail === 'string' ? payload.detail : null
        throw new Error(detail || `The request failed (HTTP ${response.status}).`)
      }
      setResult({ answer: payload.answer || '', sources: payload.sources || [], calls: payload.tool_calls || [] })
    } catch (err) {
      setResult(null)
      setError(err instanceof TypeError ? 'The API could not be reached.' : err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="shell">
      <header className="masthead">
        <p className="kicker">ONS ASHE · workplace pension tables</p>
        <h1>Ask the pension tables</h1>
        <p className="lede">
          Plain-English answers about UK workplace pension membership and contributions, each one traced back to
          the published table it came from. Statistics only, not pension advice.
        </p>
      </header>

      <main className="layout">
        <section className="card ask" aria-labelledby="ask-heading">
          <h2 id="ask-heading">Your question</h2>
          <form onSubmit={askAgent}>
            <label htmlFor="question" className="visually-hidden">Question about workplace pensions</label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={4}
              maxLength={MAX_QUESTION}
              aria-describedby="question-hint"
              placeholder="For example: how do contribution rates for small employers compare to large ones?"
            />
            <div className="form-row">
              <span id="question-hint" className="small">{trimmed.length}/{MAX_QUESTION} characters, at least {MIN_QUESTION}</span>
              <button type="submit" disabled={!canSubmit}>{loading ? 'Checking the tables…' : 'Ask'}</button>
            </div>
          </form>
          <div className="examples" aria-label="Example questions">
            {EXAMPLES.map((example) => (
              <button key={example} type="button" className="chip" onClick={() => setQuestion(example)}>
                {example}
              </button>
            ))}
          </div>
        </section>

        <aside className="card dataset" aria-labelledby="dataset-heading">
          <p className="kicker">Data in this instance</p>
          <DatasetPanel dataset={dataset} />
        </aside>

        <section className="card answer" aria-labelledby="answer-heading" aria-live="polite" aria-busy={loading}>
          <h2 id="answer-heading">Answer</h2>
          {loading ? <p className="muted">Looking up the tables and checking the draft answer…</p> : null}
          {!loading && error ? <p className="notice notice-error" role="alert">{error}</p> : null}
          {!loading && !error && !result ? (
            <p className="muted">Answers appear here with the tables they cite.</p>
          ) : null}
          {!loading && !error && result ? (
            <>
              <p className="answer-text">{result.answer || 'The assistant returned an empty answer.'}</p>
              {result.sources.length ? (
                <p className="sources"><span>Sources</span> {result.sources.join(' · ')}</p>
              ) : null}
              <ToolTrace calls={result.calls} />
            </>
          ) : null}
        </section>
      </main>
    </div>
  )
}

export default App

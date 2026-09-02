import { useMemo, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function App() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const canSubmit = useMemo(() => question.trim().length >= 5 && !loading, [question, loading])

  const askAgent = async (event) => {
    event.preventDefault()
    if (!canSubmit) return

    setLoading(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}))
        throw new Error(payload.detail || `Request failed (${response.status})`)
      }

      const payload = await response.json()
      setAnswer(payload.answer || '')
      setSources(payload.sources || [])
    } catch (err) {
      setAnswer('')
      setSources([])
      setError(err.message || 'Unexpected error while contacting the agent API.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page">
      <section className="panel">
        <form onSubmit={askAgent}>
          <label htmlFor="question">Ask a pension question</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows={4}
            placeholder="Example: How do contribution rates for small employers compare to large ones?"
          />

          <div className="actions">
            <button type="submit" disabled={!canSubmit}>
              {loading ? 'Analyzing...' : 'Ask agent'}
            </button>
          </div>
        </form>
      </section>

      <section className="panel output">
        <h2>Answer</h2>
        {error ? <p className="error">{error}</p> : null}
        {!error && answer ? <p className="answer">{answer}</p> : null}
        {!error && !answer ? <p className="empty">Submit a question to see results.</p> : null}

        {sources.length > 0 ? (
          <div>
            <h3>Sources</h3>
            <ul>
              {sources.map((source) => (
                <li key={source}>{source}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </section>
    </main>
  )
}

export default App

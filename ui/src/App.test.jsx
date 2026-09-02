import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App.jsx'

const FIXTURE_READY = {
  ready: true,
  mode: 'fixture',
  rows: 6,
  tables: ['P12', 'P2'],
  years: [2023, 2024],
  notice: 'Synthetic values for software tests; not ONS statistics.',
  agent_configured: true,
}

const ANSWER = {
  answer: 'In 2024, 65% of jobs in financial and insurance activities had a defined contribution pension.',
  sources: ['P2 (2024)'],
  tool_calls: [
    { tool: 'lookup_industry_membership_rate', ok: true, sources: ['P2 (2024)'] },
    { tool: 'explain_quality_flags', ok: false, sources: [] },
  ],
  critic: { grounded: true, advice_free: true, clear_for_hr_reader: true, issues: [] },
}

function json(body, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }))
}

function mockApi({ ready = FIXTURE_READY, ask } = {}) {
  const fetchMock = vi.fn((url) => {
    if (String(url).endsWith('/ready')) return ready instanceof Error ? Promise.reject(ready) : json(ready)
    return ask()
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('dataset panel', () => {
  it('shows a loading state, then the fixture coverage', async () => {
    mockApi()
    render(<App />)
    expect(screen.getByText('Checking the dataset…')).toBeTruthy()
    expect(await screen.findByRole('heading', { name: 'Synthetic test fixture' })).toBeTruthy()
    expect(screen.getByText('2024')).toBeTruthy()
    expect(screen.getByText(/not ONS statistics/)).toBeTruthy()
  })

  it('says when the API cannot be reached', async () => {
    mockApi({ ready: new TypeError('Failed to fetch') })
    render(<App />)
    expect((await screen.findByRole('alert')).textContent).toMatch(/did not respond/)
  })

  it('disables questions when no model is configured', async () => {
    mockApi({ ready: { ...FIXTURE_READY, agent_configured: false } })
    render(<App />)
    expect(await screen.findByText(/no model provider is configured/)).toBeTruthy()
    fireEvent.change(screen.getByLabelText(/Question about workplace pensions/), {
      target: { value: 'What is the finance rate?' },
    })
    expect(screen.getByRole('button', { name: 'Ask' }).disabled).toBe(true)
  })
})

describe('asking a question', () => {
  it('posts the trimmed question and shows the answer, sources and trace', async () => {
    const fetchMock = mockApi({ ask: () => json(ANSWER) })
    render(<App />)
    await screen.findByRole('heading', { name: 'Synthetic test fixture' })

    fireEvent.click(screen.getByRole('button', { name: /finance jobs/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }))

    expect(await screen.findByText(/65% of jobs in financial/)).toBeTruthy()
    const [, init] = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/api/ask'))
    expect(JSON.parse(init.body).question).toMatch(/^What share of finance jobs/)

    const answer = screen.getByRole('region', { name: 'Answer' })
    expect(within(answer).getByText(/P2 \(2024\)/, { selector: '.sources' })).toBeTruthy()
    expect(within(answer).getByText('How this answer was found (2 lookups)')).toBeTruthy()
    expect(within(answer).getByText('Looked up an industry membership rate')).toBeTruthy()
    expect(within(answer).getByText('no result')).toBeTruthy()
  })

  it('shows the API error detail instead of an answer', async () => {
    mockApi({ ask: () => json({ detail: 'Dataset is not ready; build a verified snapshot first.' }, 503) })
    render(<App />)
    await screen.findByRole('heading', { name: 'Synthetic test fixture' })
    fireEvent.change(screen.getByLabelText(/Question about workplace pensions/), {
      target: { value: 'What is the finance rate?' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }))
    expect((await screen.findByRole('alert')).textContent).toMatch(/build a verified snapshot/)
  })

  it('keeps the button disabled for questions under five characters', async () => {
    mockApi()
    render(<App />)
    await screen.findByRole('heading', { name: 'Synthetic test fixture' })
    fireEvent.change(screen.getByLabelText(/Question about workplace pensions/), { target: { value: '  abc ' } })
    expect(screen.getByRole('button', { name: 'Ask' }).disabled).toBe(true)
  })
})

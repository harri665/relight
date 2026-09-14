import { useEffect, useState } from 'react'

export default function Home() {
  const [health, setHealth] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then(setHealth)
      .catch((e) => setError(e.message))
  }, [])

  return (
    <section className="p-6 space-y-4">
      <h1 className="text-4xl font-bold">Home</h1>
      <div className="rounded border p-4">
        <h2 className="font-semibold mb-2">API Health</h2>
        {error && <p className="text-red-600">Error: {error}</p>}
        {health ? (
          <pre className="text-sm bg-gray-100 p-2 rounded">{JSON.stringify(health, null, 2)}</pre>
        ) : (
          <p className="text-gray-600">Loading...</p>
        )}
      </div>
    </section>
  )
}

import { Routes, Route, useLocation } from 'react-router-dom'
import Home from './pages/Home.jsx'
import StatsBar from './components/StatsBar.jsx'

function App() {
  const { pathname } = useLocation()
  return (
    <div className="flex h-full flex-col max-[800px]:h-auto">
      <header className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 border-b border-line px-4 py-2">
        {pathname === '/' && <StatsBar />}
      </header>

      <Routes>
        <Route path="/" element={<Home />} />
      </Routes>
    </div>
  )
}

export default App

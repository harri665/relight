import { NavLink, Routes, Route, useLocation } from 'react-router-dom'
import Home from './pages/Home.jsx'
import About from './pages/About.jsx'
import StatsBar from './components/StatsBar.jsx'

const link = ({ isActive }) => (isActive ? 'text-accent' : 'text-dim hover:text-fg')

function App() {
  const { pathname, search } = useLocation()
  return (
    <div className="flex h-full flex-col max-[800px]:h-auto">
      <header className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-line px-4 py-2">
        <nav className="flex items-center gap-4 text-[13px]">
          <span className="font-bold text-fg">relight</span>
          {/* keep the query (scene, backend) when switching pages */}
          <NavLink to={{ pathname: '/', search }} className={link} end>Viewer</NavLink>
          <NavLink to={{ pathname: '/about', search }} className={link}>About</NavLink>
        </nav>
        {pathname === '/' && <StatsBar />}
      </header>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </div>
  )
}

export default App

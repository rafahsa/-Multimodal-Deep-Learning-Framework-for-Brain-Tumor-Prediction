import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './theme/global.css'
import './theme/responsive.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { RnaProvider } from './context/RnaContext'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <RnaProvider>
        <App />
      </RnaProvider>
    </BrowserRouter>
  </StrictMode>,
)

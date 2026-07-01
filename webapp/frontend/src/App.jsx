import { useState, useEffect, useCallback, useRef } from 'react'
import { auth, googleProvider, signInWithPopup } from './firebase'

const API_BASE = '/api'

const CATEGORY_COLORS = {
  person: "#E8542C",   // Coral
  violin: "#3FA9C9",   // Cyan
  guitar: "#E0A540",   // Amber
  cello: "#3FA9C9",
  flute: "#E0A540",
  piano: "#3FA9C9",
  ukulele: "#E0A540",
  accordion: "#3FA9C9",
  guzheng: "#E0A540",
  clarinet: "#3FA9C9",
  cat: "#E0A540",
  car: "#3FA9C9",
  saxophone: "#E0A540",
  dog: "#3FA9C9",
  lawn_mover: "#E0A540",
  tuba: "#3FA9C9",
  banjo: "#E0A540",
  pipa: "#3FA9C9",
  bassoon: "#E0A540",
  airplane: "#3FA9C9",
  tree_harvester: "#E0A540",
  trumpet: "#3FA9C9",
  lion: "#E0A540",
  bass: "#3FA9C9",
  erhu: "#E0A540",
  horse: "#3FA9C9"
}

function App() {
  const [samples, setSamples] = useState([])
  const [selectedSample, setSelectedSample] = useState(null)
  
  const [jobStatus, setJobStatus] = useState(null) // null | 'pending' | 'running' | 'done' | 'error'
  const [jobError, setJobError] = useState(null)
  const [results, setResults] = useState(null)
  
  const [viewMode, setViewMode] = useState('video') // 'video' | 'frames'
  const [frameIndex, setFrameIndex] = useState(0)

  // Authentication State
  const [user, setUser] = useState(null) // Always start null to force auth screen on open
  const [showAuthScreen, setShowAuthScreen] = useState(true)
  const [authTransitioning, setAuthTransitioning] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  
  const [authMode, setAuthMode] = useState('signup') // Default to 'signup' as shown in the ref image
  const [authEmail, setAuthEmail] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [authName, setAuthName] = useState('')
  const [authError, setAuthError] = useState('')
  const [authSubmitting, setAuthSubmitting] = useState(false)

  const triggerTransition = (userData) => {
    setAuthTransitioning(true)
    setTimeout(() => {
      setUser(userData)
      localStorage.setItem('avis_user', JSON.stringify(userData))
      setShowAuthScreen(false)
      setAuthTransitioning(false)
    }, 400)
  }

  const handleAuthSubmit = async (e) => {
    e.preventDefault()
    setAuthError('')
    setAuthSubmitting(true)
    
    const endpoint = authMode === 'login' ? '/api/auth/login' : '/api/auth/signup'
    const body = authMode === 'login' 
      ? { email: authEmail, password: authPassword }
      : { email: authEmail, password: authPassword, displayName: authName }
      
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.error || 'Authentication failed')
      }
      
      triggerTransition(data.user)
      // Reset form
      setAuthEmail('')
      setAuthPassword('')
      setAuthName('')
    } catch (err) {
      console.error(err)
      setAuthError(err.message === 'EMAIL_EXISTS' ? 'Email already registered.' :
                   err.message === 'EMAIL_NOT_FOUND' ? 'No user found with this email.' :
                   err.message === 'INVALID_PASSWORD' ? 'Incorrect password.' : err.message)
    } finally {
      setAuthSubmitting(false)
    }
  }

  const handleGoogleSignIn = async () => {
    setAuthError('')
    setAuthSubmitting(true)
    try {
      const result = await signInWithPopup(auth, googleProvider)
      const userObj = result.user
      
      const res = await fetch('/api/auth/google', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: userObj.email,
          displayName: userObj.displayName || userObj.email.split('@')[0],
          uid: userObj.uid,
          mode: authMode
        })
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.error || 'Google Authentication failed')
      }
      
      triggerTransition(data.user)
    } catch (err) {
      console.error(err)
      setAuthError(err.message || 'Google Sign In cancelled or failed.')
    } finally {
      setAuthSubmitting(false)
    }
  }

  const handleLogout = () => {
    setAuthTransitioning(true)
    setAuthMode('login')
    setShowAuthScreen(true)
    setTimeout(() => {
      setUser(null)
      localStorage.removeItem('avis_user')
      setAuthTransitioning(false)
    }, 400)
  }
  
  // Audio/video playing state
  const [isPlaying, setIsPlaying] = useState(false)
  const [sliderPosition, setSliderPosition] = useState(50) // % position for compare slider
  const [copySuccess, setCopySuccess] = useState(false)

  // Video refs for synchronization
  const origVideoRef = useRef(null)
  const segVideoRef = useRef(null)

  const pollRef = useRef(null)
  const playIntervalRef = useRef(null)
  const isDraggingRef = useRef(false)
  const sliderContainerRef = useRef(null)

  // Intersection Observer for scroll animations
  useEffect(() => {
    const observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible')
          observer.unobserve(entry.target)
        }
      })
    }, observerOptions)

    const fadeSections = document.querySelectorAll('.section-fade')
    fadeSections.forEach(section => observer.observe(section))

    return () => {
      fadeSections.forEach(section => observer.unobserve(section))
    }
  }, [samples])

  // Load samples on mount
  useEffect(() => {
    fetch(`${API_BASE}/samples`)
      .then(r => r.json())
      .then(data => {
        const list = data.samples || []
        setSamples(list)
        if (list.length > 0) {
          setSelectedSample(list[0])
        }
      })
      .catch(err => console.error('Failed to load samples:', err))
  }, [])

  // Poll job status
  const startPolling = useCallback((sampleId) => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/status/${sampleId}`)
        const data = await r.json()
        setJobStatus(data.status)
        if (data.status === 'error') {
          setJobError(data.error || 'Inference execution halted unexpectedly.')
          clearInterval(pollRef.current)
        }
        if (data.status === 'done') {
          clearInterval(pollRef.current)
          const rr = await fetch(`${API_BASE}/results/${sampleId}`)
          const resData = await rr.json()
          setResults(resData)
          setFrameIndex(0)
        }
      } catch (e) {
        console.error('Poll error:', e)
      }
    }, 1500)
  }, [])

  // Cleanup timers
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
      if (playIntervalRef.current) clearInterval(playIntervalRef.current)
    }
  }, [])

  // Sync video play/pause status
  useEffect(() => {
    if (viewMode === 'video' && results) {
      if (isPlaying) {
        if (origVideoRef.current) origVideoRef.current.play().catch(() => {})
        if (segVideoRef.current) segVideoRef.current.play().catch(() => {})
      } else {
        if (origVideoRef.current) origVideoRef.current.pause()
        if (segVideoRef.current) segVideoRef.current.pause()
      }
    }
  }, [isPlaying, viewMode, results])

  // Frame player animation (only for frames view mode)
  useEffect(() => {
    if (isPlaying && results && viewMode === 'frames') {
      playIntervalRef.current = setInterval(() => {
        setFrameIndex((prev) => (prev + 1) % results.num_frames)
      }, 100)
    } else {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current)
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current)
    }
  }, [isPlaying, results, viewMode])

  const handleSelectSample = (sample) => {
    setSelectedSample(sample)
    setJobStatus(null)
    setJobError(null)
    setResults(null)
    setFrameIndex(0)
    setIsPlaying(false)
    if (pollRef.current) clearInterval(pollRef.current)
  }

  const handleRunInference = async () => {
    if (!selectedSample) return
    if (!user) {
      setAuthMode('login')
      setShowAuthScreen(true)
      return
    }
    setJobStatus('pending')
    setJobError(null)
    setResults(null)
    setIsPlaying(false)

    try {
      const r = await fetch(`${API_BASE}/infer/${selectedSample.id}`, { method: 'POST' })
      const data = await r.json()
      setJobStatus(data.status)

      if (data.status === 'done') {
        const rr = await fetch(`${API_BASE}/results/${selectedSample.id}`)
        const resData = await rr.json()
        setResults(resData)
        setFrameIndex(0)
      } else {
        startPolling(selectedSample.id)
      }
    } catch (e) {
      setJobStatus('error')
      setJobError('Failed to establish connection with the inference server. Verify that the Flask backend is active.')
    }
  }

  // Slider Mouse events
  const handleMouseMove = (e) => {
    if (!isDraggingRef.current || !sliderContainerRef.current) return
    const rect = sliderContainerRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setSliderPosition(percentage)
  }

  const handleMouseUp = () => {
    isDraggingRef.current = false
    window.removeEventListener('mousemove', handleMouseMove)
    window.removeEventListener('mouseup', handleMouseUp)
  }

  const handleMouseDown = () => {
    isDraggingRef.current = true
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
  }

  const copyToClipboard = () => {
    const text = `@article{guo2023audio,
  title={Audio-Visual Instance Segmentation},
  author={Guo, Ruohao and Ying, Xianghua and Chen, Yaru and Niu, Dantong and Li, Guangyao and Qu, Liao and Qi, Yanyu and Zhou, Jinxing and Xing, Bowei and Yue, Wenzhen and Shi, Ji and Wang, Qixun and Zhang, Peiliang and Liang, Buwen},
  journal={arXiv preprint arXiv:2310.18709},
  year={2023}
}`
    navigator.clipboard.writeText(text)
      .then(() => {
        setCopySuccess(true)
        setTimeout(() => setCopySuccess(false), 2000)
      })
  }

  const getAudioWavePath = (sampleId, numBars = 50) => {
    if (!sampleId) return ""
    let seed = 0
    for (let i = 0; i < sampleId.length; i++) seed += sampleId.charCodeAt(i)
    const random = () => {
      const x = Math.sin(seed++) * 10000
      return x - Math.floor(x)
    }

    let path = ""
    const barWidth = 3
    const gap = 2
    const height = 20
    for (let i = 0; i < numBars; i++) {
      const barHeight = 2 + random() * (height - 4)
      const x = i * (barWidth + gap)
      const y = (height - barHeight) / 2
      path += `M${x},${y} L${x},${y + barHeight} `
    }
    return path
  }

  return (
    <div className="relative min-h-screen bg-[#08090a] overflow-x-hidden">
      
      {/* Main Website App Container with Premium Fade & Scale Up transition */}
      <div 
        className={`custom-scrollbar transition-all duration-500 ease-out ${
          showAuthScreen ? 'opacity-0 scale-95 pointer-events-none' : 'opacity-100 scale-100'
        }`}
      >
      
      {/* Top Navigation Bar */}
      <nav className="fixed top-0 left-0 w-full z-50 border-b border-surface-container-highest bg-background/80 backdrop-blur-md">
        <div className="flex justify-between items-center w-full px-margin-mobile md:px-margin-desktop max-w-max-width mx-auto h-20">
          <div className="font-headline-md text-headline-md font-bold text-on-background tracking-tight flex items-center gap-2">
            <span className="material-symbols-outlined text-primary text-[28px]">biotech</span>
            AVIS
          </div>
          <div className="hidden md:flex gap-8 items-center font-body-md text-body-md uppercase tracking-wider">
            <a className="text-primary border-b-2 border-primary pb-1 transition-all duration-[250ms] ease-out" href="#overview">Overview</a>
            <a className="text-on-surface-variant hover:text-primary border-b-2 border-transparent hover:border-primary/40 pb-1 transition-all duration-[250ms] ease-out" href="#results">Results</a>
            <a className="text-on-surface-variant hover:text-primary border-b-2 border-transparent hover:border-primary/40 pb-1 transition-all duration-[250ms] ease-out" href="#demo">Demo</a>
            <a className="text-on-surface-variant hover:text-primary border-b-2 border-transparent hover:border-primary/40 pb-1 transition-all duration-[250ms] ease-out" href="#authors">Team</a>
          </div>
          <div className="flex items-center gap-6">
            <a className="material-symbols-outlined text-on-surface-variant hover:text-primary transition-all duration-[250ms] ease-out hover:-translate-y-[2px]" href="#citation">description</a>
            {user ? (
              <div className="flex items-center gap-3">
                <span className="font-label-sm text-[12px] bg-[#1B1F24] px-3 py-1.5 border border-[rgba(255,255,255,0.08)] rounded-[12px] text-on-surface-variant flex items-center gap-1">
                  <span className="material-symbols-outlined text-[14px]">person</span>
                  {user.displayName}
                </span>
                <button 
                  onClick={handleLogout} 
                  className="bg-transparent text-primary hover:text-white hover:bg-primary border border-primary px-4 py-1.5 rounded-[18px] text-label-sm font-label-sm uppercase hover:-translate-y-[2px] transition-all duration-[250ms] ease-out"
                >
                  Logout
                </button>
              </div>
            ) : (
              <button 
                onClick={() => { setAuthMode('login'); setShowAuthScreen(true); }}
                className="bg-primary text-white px-4 py-1.5 border border-primary rounded-[18px] text-label-sm font-label-sm uppercase hover:-translate-y-[2px] transition-all duration-[250ms] ease-out flex items-center gap-1.5"
              >
                <span className="material-symbols-outlined text-[16px]">login</span>
                Sign In
              </button>
            )}
            <button className="bg-primary-container text-white px-6 py-2 border border-primary rounded-[18px] text-label-sm font-label-sm uppercase hover:-translate-y-[2px] transition-all duration-[250ms] ease-out flex items-center gap-2">
              <span className="material-symbols-outlined text-[16px]">picture_as_pdf</span>
              Paper PDF
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="relative h-screen w-full flex flex-col justify-end overflow-hidden">
        <div className="absolute inset-0 z-0">
          <div 
            className="w-full h-full bg-cover bg-center" 
            style={{ backgroundImage: `url('/highres_cover_image.png')` }}
          ></div>
          <div className="absolute inset-0 bg-gradient-to-t from-background via-background/40 to-transparent"></div>
        </div>
        <div className="relative z-10 w-full max-w-max-width mx-auto px-margin-mobile md:px-margin-desktop pb-24 hero-fade-in">
          
          <div className="w-full h-12 mb-8 opacity-60">
            <svg className="w-full h-full stroke-primary fill-none stroke-[2]" preserveAspectRatio="none" viewBox="0 0 1000 100">
              <path d="M0 50 Q 25 10, 50 50 T 100 50 T 150 50 T 200 50 T 250 50 T 300 50 T 350 50 T 400 50 T 450 50 T 500 50 T 550 50 T 600 50 T 650 50 T 700 50 T 750 50 T 800 50 T 850 50 T 900 50 T 950 50 T 1000 50">
                <animate 
                  attributeName="d" 
                  dur="5s" 
                  repeatCount="indefinite" 
                  values="M0 50 Q 25 10, 50 50 T 100 50 T 150 50 T 200 50 T 250 50 T 300 50 T 350 50 T 400 50 T 450 50 T 500 50 T 550 50 T 600 50 T 650 50 T 700 50 T 750 50 T 800 50 T 850 50 T 900 50 T 950 50 T 1000 50; M0 50 Q 25 90, 50 50 T 100 50 T 150 50 T 200 50 T 250 50 T 300 50 T 350 50 T 400 50 T 450 50 T 500 50 T 550 50 T 600 50 T 650 50 T 700 50 T 750 50 T 800 50 T 850 50 T 900 50 T 950 50 T 1000 50; M0 50 Q 25 10, 50 50 T 100 50 T 150 50 T 200 50 T 250 50 T 300 50 T 350 50 T 400 50 T 450 50 T 500 50 T 550 50 T 600 50 T 650 50 T 700 50 T 750 50 T 800 50 T 850 50 T 900 50 T 950 50 T 1000 50"
                />
              </path>
            </svg>
          </div>

          <h1 className="font-headline-lg text-[56px] sm:text-[80px] md:text-[140px] uppercase font-bold tracking-tighter leading-[0.85] mb-6 flex items-baseline gap-4">
            AVIS
          </h1>
          <p className="font-body-lg text-body-lg text-on-surface-variant max-w-xl mb-10 border-l-2 border-primary pl-6 flex items-center gap-2">
            <span className="material-symbols-outlined text-primary text-[20px]">hearing</span>
            Segmenting objects using both sound and sight.
          </p>
          <a href="#demo" className="bg-primary-container text-white px-12 py-4 font-label-sm text-label-sm uppercase tracking-widest border border-primary rounded-[18px] hover:-translate-y-[2px] transition-all duration-[250ms] ease-out inline-block text-center shadow-md">
            Try It
          </a>
        </div>
      </header>

      {/* Main Content Grid */}
      <main className="w-full max-w-max-width mx-auto px-margin-mobile md:px-margin-desktop space-y-24 md:space-y-32 py-16 md:py-32">
        
        {/* Overview Section */}
        <section className="section-fade space-y-12" id="overview">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-gutter items-start">
            <div className="md:col-span-4">
              <span className="font-label-sm text-label-sm text-primary uppercase tracking-widest mb-4 block flex items-center gap-2">
                <span className="material-symbols-outlined text-[16px]">info</span>
                01 / Motivation
              </span>
              <h2 className="font-headline-md text-headline-md uppercase mb-8">Technical Overview</h2>
              <a className="inline-flex items-center gap-2 text-primary font-label-sm uppercase tracking-widest hover:gap-4 transition-all" href="#">
                Read Paper <span className="material-symbols-outlined text-[18px]">arrow_forward</span>
              </a>
            </div>
            
            <div className="md:col-span-8 border-l-0 md:border-l border-surface-container-highest pl-0 md:pl-12 mt-8 md:mt-0">
              <p className="font-body-lg text-body-lg text-on-surface leading-relaxed mb-12">
                Audio-Visual Instance Segmentation (AVIS) extends traditional vision-only segmentation by leveraging temporal acoustic cues to disambiguate occluded or visually similar objects. Our framework synchronizes multi-modal embeddings to isolate specific instances, significantly outperforming vision-exclusive models in complex dynamic environments.
              </p>
              
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
                <div className="space-y-1">
                  <div className="font-code-md text-primary text-xl flex items-center gap-1">
                    <span className="material-symbols-outlined text-[18px]">video_library</span>
                    50k+
                  </div>
                  <div className="font-label-sm text-[10px] uppercase opacity-60">Frames processed</div>
                </div>
                <div className="space-y-1">
                  <div className="font-code-md text-secondary text-xl flex items-center gap-1">
                    <span className="material-symbols-outlined text-[18px]">category</span>
                    26
                  </div>
                  <div className="font-label-sm text-[10px] uppercase opacity-60">Categories</div>
                </div>
                <div className="space-y-1">
                  <div className="font-code-md text-tertiary text-xl flex items-center gap-1">
                    <span className="material-symbols-outlined text-[18px]">bolt</span>
                    seconds
                  </div>
                  <div className="font-label-sm text-[10px] uppercase opacity-60">Inference</div>
                </div>
              </div>
            </div>
          </div>

          {/* System Architecture Box */}
          <div className="w-full border border-[rgba(255,255,255,0.08)] bg-[#1B1F24] p-8 rounded-[24px] overflow-hidden">
            <div className="font-label-sm text-[10px] uppercase text-on-surface-variant/60 mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-[14px]">account_tree</span>
              System Architecture
            </div>
            <img 
              alt="AVIS Architecture Diagram" 
              className="w-full h-auto block rounded-[16px]" 
              src="/arch.png" 
            />
          </div>
        </section>

        {/* Results Section */}
        <section className="section-fade space-y-12" id="results">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
            <div>
              <span className="font-label-sm text-label-sm text-primary uppercase tracking-widest mb-4 block flex items-center gap-2">
                <span className="material-symbols-outlined text-[16px]">equalizer</span>
                02 / Benchmarks
              </span>
              <h2 className="font-headline-md text-headline-md uppercase">Performance Metrics</h2>
            </div>
            <div className="flex gap-4">
              <span className="flex items-center gap-2 font-label-sm text-label-sm text-secondary border border-secondary px-3 py-1">
                <span className="material-symbols-outlined text-[14px]">query_stats</span>
                mAP: 40.57
              </span>
              <span className="flex items-center gap-2 font-label-sm text-label-sm text-tertiary border border-tertiary px-3 py-1">
                <span className="material-symbols-outlined text-[14px]">grid_view</span>
                HOTA: 61.73
              </span>
            </div>
          </div>

          <div className="overflow-x-auto border border-surface-container-highest bg-surface-container-lowest">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface-container-high border-b border-surface-container-highest text-left">
                  <th className="px-4 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant">Task</th>
                  <th className="px-4 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant">Model</th>
                  <th className="px-4 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant">Venue</th>
                  <th className="px-4 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant text-center">Audio</th>
                  <th className="px-3 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant text-center">FSLA</th>
                  <th className="px-3 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant text-center">HOTA</th>
                  <th className="px-3 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant text-center">mAP</th>
                  <th className="px-3 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant text-center">FSLAn</th>
                  <th className="px-3 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant text-center">FSLAs</th>
                  <th className="px-3 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant text-center">FSLAm</th>
                  <th className="px-3 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant text-center">AssA</th>
                  <th className="px-3 py-4 font-label-sm text-[11px] uppercase text-on-surface-variant text-center">DetA</th>
                </tr>
              </thead>
              <tbody className="font-code-md text-code-md divide-y divide-surface-container-highest">
                <tr className="hover:bg-surface-container transition-colors group">
                  <td className="px-4 py-3.5 text-on-surface-variant">VIS</td>
                  <td className="px-4 py-3.5">Mask2Former-VIS [11]</td>
                  <td className="px-4 py-3.5">CVPR' 22</td>
                  <td className="px-4 py-3.5 text-center text-error">✘</td>
                  <td className="px-3 py-3.5 text-center">29.75</td>
                  <td className="px-3 py-3.5 text-center">52.03</td>
                  <td className="px-3 py-3.5 text-center">28.66</td>
                  <td className="px-3 py-3.5 text-center text-error">0.00</td>
                  <td className="px-3 py-3.5 text-center">25.47</td>
                  <td className="px-3 py-3.5 text-center">36.37</td>
                  <td className="px-3 py-3.5 text-center">64.49</td>
                  <td className="px-3 py-3.5 text-center">43.33</td>
                </tr>
                <tr className="hover:bg-surface-container transition-colors group">
                  <td className="px-4 py-3.5 text-on-surface-variant">VIS</td>
                  <td className="px-4 py-3.5">TeVIT [58]</td>
                  <td className="px-4 py-3.5">CVPR' 22</td>
                  <td className="px-4 py-3.5 text-center text-error">✘</td>
                  <td className="px-3 py-3.5 text-center">32.28</td>
                  <td className="px-3 py-3.5 text-center">53.67</td>
                  <td className="px-3 py-3.5 text-center">31.52</td>
                  <td className="px-3 py-3.5 text-center text-error">0.00</td>
                  <td className="px-3 py-3.5 text-center">28.07</td>
                  <td className="px-3 py-3.5 text-center">39.18</td>
                  <td className="px-3 py-3.5 text-center">65.27</td>
                  <td className="px-3 py-3.5 text-center">45.10</td>
                </tr>
                <tr className="hover:bg-surface-container transition-colors group">
                  <td className="px-4 py-3.5 text-on-surface-variant">VIS</td>
                  <td className="px-4 py-3.5">SeqFormer [52]</td>
                  <td className="px-4 py-3.5">ECCV' 22</td>
                  <td className="px-4 py-3.5 text-center text-error">✘</td>
                  <td className="px-3 py-3.5 text-center">30.32</td>
                  <td className="px-3 py-3.5 text-center">54.32</td>
                  <td className="px-3 py-3.5 text-center">32.79</td>
                  <td className="px-3 py-3.5 text-center">25.03</td>
                  <td className="px-3 py-3.5 text-center">21.76</td>
                  <td className="px-3 py-3.5 text-center">36.46</td>
                  <td className="px-3 py-3.5 text-center">67.25</td>
                  <td className="px-3 py-3.5 text-center">45.23</td>
                </tr>
                <tr className="hover:bg-surface-container transition-colors group">
                  <td className="px-4 py-3.5 text-on-surface-variant">VIS</td>
                  <td className="px-4 py-3.5">VITA [26]</td>
                  <td className="px-4 py-3.5">NeurIPS' 22</td>
                  <td className="px-4 py-3.5 text-center text-error">✘</td>
                  <td className="px-3 py-3.5 text-center">38.04</td>
                  <td className="px-3 py-3.5 text-center">57.48</td>
                  <td className="px-3 py-3.5 text-center">36.25</td>
                  <td className="px-3 py-3.5 text-center">15.04</td>
                  <td className="px-3 py-3.5 text-center">27.98</td>
                  <td className="px-3 py-3.5 text-center">47.45</td>
                  <td className="px-3 py-3.5 text-center">69.86</td>
                  <td className="px-3 py-3.5 text-center">48.96</td>
                </tr>
                <tr className="hover:bg-surface-container transition-colors group">
                  <td className="px-4 py-3.5 text-on-surface-variant">VIS</td>
                  <td className="px-4 py-3.5">DAVIS [60]</td>
                  <td className="px-4 py-3.5">ICCV' 23</td>
                  <td className="px-4 py-3.5 text-center text-error">✘</td>
                  <td className="px-3 py-3.5 text-center">23.99</td>
                  <td className="px-3 py-3.5 text-center">49.12</td>
                  <td className="px-3 py-3.5 text-center">19.83</td>
                  <td className="px-3 py-3.5 text-center">14.61</td>
                  <td className="px-3 py-3.5 text-center">24.83</td>
                  <td className="px-3 py-3.5 text-center">24.69</td>
                  <td className="px-3 py-3.5 text-center">63.51</td>
                  <td className="px-3 py-3.5 text-center">40.11</td>
                </tr>
                <tr className="hover:bg-surface-container transition-colors group">
                  <td className="px-4 py-3.5 text-on-surface-variant">VIS</td>
                  <td className="px-4 py-3.5">LBVQ [16]</td>
                  <td className="px-4 py-3.5">TCSVT' 24</td>
                  <td className="px-4 py-3.5 text-center text-error">✘</td>
                  <td className="px-3 py-3.5 text-center">34.73</td>
                  <td className="px-3 py-3.5 text-center">56.97</td>
                  <td className="px-3 py-3.5 text-center">36.58</td>
                  <td className="px-3 py-3.5 text-center">27.71</td>
                  <td className="px-3 py-3.5 text-center">29.52</td>
                  <td className="px-3 py-3.5 text-center">38.96</td>
                  <td className="px-3 py-3.5 text-center">68.34</td>
                  <td className="px-3 py-3.5 text-center">48.83</td>
                </tr>

                <tr className="hover:bg-surface-container transition-colors group">
                  <td className="px-4 py-3.5 text-on-surface-variant">AVSS</td>
                  <td className="px-4 py-3.5">AVSegFormer [17]</td>
                  <td className="px-4 py-3.5">AAAI' 24</td>
                  <td className="px-4 py-3.5 text-center text-secondary">✔</td>
                  <td className="px-3 py-3.5 text-center">35.66</td>
                  <td className="px-3 py-3.5 text-center">55.74</td>
                  <td className="px-3 py-3.5 text-center">35.72</td>
                  <td className="px-3 py-3.5 text-center">18.58</td>
                  <td className="px-3 py-3.5 text-center">27.51</td>
                  <td className="px-3 py-3.5 text-center">43.08</td>
                  <td className="px-3 py-3.5 text-center">67.13</td>
                  <td className="px-3 py-3.5 text-center">48.51</td>
                </tr>
                <tr className="hover:bg-surface-container transition-colors group">
                  <td className="px-4 py-3.5 text-on-surface-variant">AVSS</td>
                  <td className="px-4 py-3.5">COMBO [56]</td>
                  <td className="px-4 py-3.5">CVPR' 24</td>
                  <td className="px-4 py-3.5 text-center text-secondary">✔</td>
                  <td className="px-3 py-3.5 text-center">39.49</td>
                  <td className="px-3 py-3.5 text-center">57.39</td>
                  <td className="px-3 py-3.5 text-center">37.84</td>
                  <td className="px-3 py-3.5 text-center">21.91</td>
                  <td className="px-3 py-3.5 text-center">27.18</td>
                  <td className="px-3 py-3.5 text-center">49.63</td>
                  <td className="px-3 py-3.5 text-center">68.87</td>
                  <td className="px-3 py-3.5 text-center">50.12</td>
                </tr>
                
                {/* Highlighted ours row */}
                <tr className="bg-primary/5 border-l-4 border-primary hover:bg-primary/10 transition-colors">
                  <td className="px-4 py-3.5 text-primary font-bold">AVIS</td>
                  <td className="px-4 py-3.5 font-bold text-primary">AVISM</td>
                  <td className="px-4 py-3.5 font-bold">CVPR' 25</td>
                  <td className="px-4 py-3.5 text-center text-secondary font-bold">✔</td>
                  <td className="px-3 py-3.5 text-center font-bold">42.78</td>
                  <td className="px-3 py-3.5 text-center font-bold">61.73</td>
                  <td className="px-3 py-3.5 text-center font-bold">40.57</td>
                  <td className="px-3 py-3.5 text-center font-bold">32.22</td>
                  <td className="px-3 py-3.5 text-center font-bold">29.83</td>
                  <td className="px-3 py-3.5 text-center font-bold">52.40</td>
                  <td className="px-3 py-3.5 text-center font-bold">71.15</td>
                  <td className="px-3 py-3.5 text-center font-bold">54.97</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Demo Section */}
        <section className="section-fade space-y-12" id="demo">
          <div>
            <span className="font-label-sm text-label-sm text-primary uppercase tracking-widest mb-4 block flex items-center gap-2">
              <span className="material-symbols-outlined text-[16px]">play_circle</span>
              03 / Inference
            </span>
            <h2 className="font-headline-md text-headline-md uppercase">Interactive Demo</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-gutter">
            
            {/* Sidebar sample picker */}
            <div className="lg:col-span-1 flex flex-row lg:flex-col overflow-x-auto lg:overflow-y-auto max-h-[160px] lg:max-h-[500px] gap-4 lg:space-y-4 pr-0 lg:pr-2 pb-4 lg:pb-0 custom-scrollbar">
              {samples.map(sample => (
                <div
                  key={sample.id}
                  onClick={() => handleSelectSample(sample)}
                  className={`border p-1.5 cursor-pointer rounded-[20px] transition-all duration-[250ms] ease-out hover:-translate-y-[2px] min-w-[150px] sm:min-w-[180px] lg:min-w-0 ${
                    selectedSample?.id === sample.id 
                      ? 'border-primary bg-primary/5 opacity-100' 
                      : 'border-[rgba(255,255,255,0.08)] opacity-60 hover:opacity-100 hover:bg-[#1B1F24]'
                  }`}
                >
                  <div className="aspect-video bg-cover bg-center rounded-[14px]" style={{ backgroundImage: `url('${API_BASE}/samples/${sample.id}/thumbnail')` }}></div>
                  <div className={`p-2 font-label-sm text-[10px] uppercase flex justify-between items-center ${
                    selectedSample?.id === sample.id ? 'text-primary font-bold' : 'text-on-surface-variant'
                  }`}>
                    <span className="truncate max-w-[80%]">{sample.name}</span>
                    {selectedSample?.id === sample.id && (
                      <span className="material-symbols-outlined text-[14px]">check_circle</span>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Main Viewport */}
            <div className="lg:col-span-3 space-y-6">
              
              {/* Top Viewport Header */}
              {selectedSample && (
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-surface-container-highest pb-4">
                  <div className="space-y-1">
                    <h3 className="font-headline-md text-lg text-on-background uppercase">{selectedSample.name}</h3>
                    <span className="text-[11px] font-code-md text-on-surface-variant uppercase flex items-center gap-3">
                      <span>{selectedSample.width}×{selectedSample.height}px</span>
                      <span>•</span>
                      <span>{selectedSample.num_frames} frames</span>
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-4">
                    {results && (
                      <div className="flex border border-[rgba(255,255,255,0.08)] bg-[#15181D] p-[3px] rounded-[18px]">
                        <button
                          onClick={() => { setViewMode('video'); setIsPlaying(false); }}
                          className={`px-4 py-1.5 rounded-[14px] text-[11px] font-label-sm uppercase transition-all duration-[250ms] ease-out flex items-center gap-1 ${
                            viewMode === 'video' ? 'bg-[#1B1F24] text-primary shadow-sm' : 'text-on-surface-variant hover:text-white'
                          }`}
                        >
                          <span className="material-symbols-outlined text-[12px]">movie</span>
                          Video Comparison
                        </button>
                        <button
                          onClick={() => { setViewMode('frames'); setIsPlaying(false); }}
                          className={`px-4 py-1.5 rounded-[14px] text-[11px] font-label-sm uppercase transition-all duration-[250ms] ease-out flex items-center gap-1 ${
                            viewMode === 'frames' ? 'bg-[#1B1F24] text-primary shadow-sm' : 'text-on-surface-variant hover:text-white'
                          }`}
                        >
                          <span className="material-symbols-outlined text-[12px]">filter</span>
                          Frame Comparison
                        </button>
                      </div>
                    )}
                    <button
                      onClick={handleRunInference}
                      disabled={jobStatus === 'pending' || jobStatus === 'running'}
                      className="bg-primary-container text-white px-6 py-2.5 border border-primary rounded-[18px] text-label-sm font-label-sm uppercase hover:-translate-y-[2px] transition-all duration-[250ms] ease-out disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center gap-2"
                    >
                      <span className="material-symbols-outlined text-[14px]">play_arrow</span>
                      {jobStatus === 'pending' || jobStatus === 'running' ? 'Running...' : 'Run inference'}
                    </button>
                  </div>
                </div>
              )}

              {/* Main media container */}
              <div className="relative border border-[rgba(255,255,255,0.08)] bg-[#15181D] rounded-[24px] overflow-hidden min-h-[300px] flex flex-col justify-center">
                
                {/* 1. Inference state simulation / Loading overlay */}
                {(jobStatus === 'pending' || jobStatus === 'running') && (
                  <div className="absolute inset-0 bg-background/90 backdrop-blur-sm flex flex-col items-center justify-center z-20">
                    <div className="w-24 h-1 border border-surface-container-highest bg-surface-container-high relative overflow-hidden mb-6">
                      <div className="absolute inset-0 bg-primary w-1/3 animate-[translateX_1s_infinite_linear]"></div>
                    </div>
                    <span className="font-code-md text-code-md text-primary uppercase tracking-[0.2em] loading-text flex items-center gap-2">
                      <span className="material-symbols-outlined animate-spin text-[16px]">sync</span>
                    </span>
                  </div>
                )}

                {/* 2. Error state overlay */}
                {jobStatus === 'error' && (
                  <div className="absolute inset-0 bg-background/95 flex flex-col items-center justify-center gap-4 p-8 text-center z-20">
                    <span className="material-symbols-outlined text-error text-[48px]">warning</span>
                    <p className="font-body-md text-on-surface max-w-md">{jobError}</p>
                    <button 
                      onClick={handleRunInference} 
                      className="bg-primary-container text-on-primary-container px-6 py-2 border border-primary text-label-sm font-label-sm uppercase hover:brightness-110 transition-all flex items-center gap-1"
                    >
                      <span className="material-symbols-outlined text-[16px]">replay</span>
                      Retry
                    </button>
                  </div>
                )}

                {/* 3. Invitation state */}
                {!jobStatus && !results && selectedSample && (
                  <div className="absolute inset-0 bg-background/80 flex flex-col items-center justify-center text-center p-8 z-10">
                    <span className="material-symbols-outlined text-primary text-[36px] mb-3">multimodal</span>
                    <p className="font-body-md text-on-surface-variant max-w-sm mb-4">
                      Select a sample and click "Run inference" to segment active sources using audio and video.
                    </p>
                    <div className="flex gap-2 justify-center">
                      {selectedSample.categories.map(c => (
                        <span 
                          key={c} 
                          className="text-[10px] font-label-sm uppercase border border-surface-container-highest bg-surface-container/30 px-3 py-1 text-on-surface-variant"
                        >
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* 4. Display: Video Mode with Synchronized Playback */}
                {results && viewMode === 'video' && (
                  <div className="flex flex-col">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-[1px] bg-surface-container-highest">
                      <div className="relative bg-surface-container-lowest">
                        <video
                          ref={origVideoRef}
                          key={`orig-${selectedSample.id}`}
                          loop
                          muted
                          playsInline
                          className="w-full aspect-video object-contain"
                          src={`${API_BASE}/results/${selectedSample.id}/original_video?t=${Date.now()}`}
                          onPlay={() => setIsPlaying(true)}
                          onPause={() => setIsPlaying(false)}
                        />
                        <span className="absolute top-4 left-4 font-label-sm text-[10px] uppercase bg-background/90 px-3 py-1 border border-surface-container-highest flex items-center gap-1 z-10">
                          <span className="material-symbols-outlined text-[10px]">videocam</span>
                          Original Input
                        </span>
                      </div>
                      <div className="relative bg-surface-container-lowest">
                        <video
                          ref={segVideoRef}
                          key={`seg-${selectedSample.id}`}
                          loop
                          muted
                          playsInline
                          className="w-full aspect-video object-contain"
                          src={`${API_BASE}/results/${selectedSample.id}/video?t=${Date.now()}`}
                          onPlay={() => setIsPlaying(true)}
                          onPause={() => setIsPlaying(false)}
                        />
                        <span className="absolute top-4 left-4 font-label-sm text-[10px] uppercase bg-primary-container text-on-primary-container px-3 py-1 border border-primary/50 flex items-center gap-1 z-10">
                          <span className="material-symbols-outlined text-[10px]">masks</span>
                          AVISM Segmented Output
                        </span>
                      </div>
                    </div>

                    {/* Unified Playback control bar for videos */}
                    <div className="flex items-center gap-4 bg-surface-container/30 p-4 border-t border-surface-container-highest">
                      <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className={`p-2 border border-surface-container-highest hover:bg-surface-container flex items-center justify-center transition-all ${
                          isPlaying ? 'text-primary' : 'text-on-surface-variant'
                        }`}
                      >
                        <span className="material-symbols-outlined text-[18px]">
                          {isPlaying ? 'pause' : 'play_arrow'}
                        </span>
                      </button>
                      <span className="font-label-sm text-[10px] uppercase tracking-widest text-on-surface-variant">
                        {isPlaying ? 'Playing Synchronized Videos' : 'Videos Paused (Click play to synchronize)'}
                      </span>
                    </div>
                  </div>
                )}

                {/* 5. Display: Interactive Frame-by-Frame Comparison Slider */}
                {results && viewMode === 'frames' && results.num_frames > 0 && (
                  <div className="flex flex-col">
                    <div
                      className="swipe-slider-container select-none"
                      ref={sliderContainerRef}
                      style={{ aspectRatio: `${selectedSample.width}/${selectedSample.height}` }}
                    >
                      {/* Underlay: Original image */}
                      <div className="slider-pane original">
                        <img
                          src={`${API_BASE}/samples/${selectedSample.id}/original_frame/${results.frame_names[frameIndex]}`}
                          alt="Original frame"
                          draggable="false"
                        />
                        <div className="slider-tag left flex items-center gap-1">
                          <span className="material-symbols-outlined text-[10px]">image</span>
                          Original Frame
                        </div>
                      </div>

                      {/* Overlay: Segmented image */}
                      <div
                        className="slider-pane segmented"
                        style={{ clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)` }}
                      >
                        <img
                          src={`${API_BASE}/results/${selectedSample.id}/frame/${results.frame_names[frameIndex]}`}
                          alt="Segmented frame"
                          draggable="false"
                        />
                        <div className="slider-tag right flex items-center gap-1">
                          <span className="material-symbols-outlined text-[10px]">masks</span>
                          AVISM Masked Frame
                        </div>
                      </div>

                      {/* Slider Handle */}
                      <div
                        className="slider-handle"
                        style={{ left: `${sliderPosition}%` }}
                        onMouseDown={handleMouseDown}
                      >
                        <div className="handle-line"></div>
                        <div className="handle-knob">
                          <span className="material-symbols-outlined text-[14px]">unfold_more</span>
                        </div>
                      </div>
                    </div>

                    {/* Scrubber Timeline controls */}
                    <div className="flex items-center gap-4 bg-surface-container/30 p-4 border-t border-surface-container-highest">
                      <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className={`p-2 border border-surface-container-highest hover:bg-surface-container flex items-center justify-center transition-all ${
                          isPlaying ? 'text-primary' : 'text-on-surface-variant'
                        }`}
                      >
                        <span className="material-symbols-outlined text-[16px]">
                          {isPlaying ? 'pause' : 'play_arrow'}
                        </span>
                      </button>

                      <div className="flex-1 flex flex-col gap-1">
                        <input
                          type="range"
                          min={0}
                          max={results.num_frames - 1}
                          value={frameIndex}
                          onChange={e => {
                            setFrameIndex(parseInt(e.target.value))
                            setIsPlaying(false)
                          }}
                          className="w-full accent-primary h-1 bg-surface-container-highest outline-none cursor-pointer"
                        />
                        <div className="flex justify-between text-[10px] font-code-md text-on-surface-variant/60 uppercase">
                          <span className="flex items-center gap-1">
                            <span className="material-symbols-outlined text-[12px]">skip_next</span>
                            Scrub Timeline Frame-by-Frame
                          </span>
                          <span>Frame {frameIndex + 1} / {results.num_frames}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

              </div>

              {/* Waveform strip and detected instances always aligned below */}
              {selectedSample && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-gutter border-t border-surface-container-highest pt-6">
                  
                  {/* Waveform strip */}
                  <div className="space-y-2">
                    <div className="font-label-sm text-[10px] uppercase text-on-surface-variant/60 flex items-center gap-1">
                      <span className="material-symbols-outlined text-[12px] text-primary">waves</span>
                      Aligned Acoustic Signal (128-D VGGish)
                    </div>
                    <div className="bg-[#1B1F24] border border-[rgba(255,255,255,0.08)] rounded-[18px] p-3 flex items-center h-[52px]">
                      <svg className="w-full h-full" viewBox="0 0 250 20">
                        <path
                          d={getAudioWavePath(selectedSample.id)}
                          fill="none"
                          stroke={isPlaying || jobStatus === 'running' ? "#70d3f5" : "#fabc54"}
                          strokeWidth="2"
                          strokeLinecap="round"
                        />
                      </svg>
                    </div>
                  </div>

                  {/* Detected objects summary */}
                  <div className="space-y-2">
                    <div className="font-label-sm text-[10px] uppercase text-on-surface-variant/60 flex items-center gap-1">
                      <span className="material-symbols-outlined text-[12px] text-primary">label</span>
                      Segment Categories
                    </div>
                    <div className="bg-[#1B1F24] border border-[rgba(255,255,255,0.08)] rounded-[18px] p-3 flex items-center flex-wrap gap-2 min-h-[52px]">
                      {results ? (
                        results.detections.length === 0 ? (
                          <span className="font-code-md text-[11px] text-on-surface-variant/40 uppercase">No active segments detected</span>
                        ) : (
                          results.detections.map((det, idx) => {
                            const color = CATEGORY_COLORS[det.category] || "#ffb4a1"
                            return (
                              <div
                                key={idx}
                                className="flex items-center gap-2 border px-3 py-1 font-label-sm text-[10px] uppercase"
                                style={{
                                  borderColor: `${color}40`,
                                  backgroundColor: `${color}0D`,
                                  color: color
                                }}
                              >
                                <span className="w-[6px] h-[6px] rounded-full" style={{ backgroundColor: color }}></span>
                                <span>{det.category}</span>
                                <span className="opacity-70 font-code-md">{(det.score * 100).toFixed(0)}%</span>
                              </div>
                            )
                          })
                        )
                      ) : (
                        <div className="flex gap-2">
                          {selectedSample.categories.map(c => (
                            <span 
                              key={c}
                              className="text-[10px] font-label-sm uppercase border border-dashed border-surface-container-highest text-on-surface-variant/40 px-2 py-0.5"
                            >
                              {c}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                </div>
              )}

            </div>

          </div>
        </section>

        {/* Authors Section */}
        <section className="section-fade space-y-12" id="authors">
          <div className="text-center max-w-2xl mx-auto">
            <span className="font-label-sm text-label-sm text-primary uppercase tracking-widest mb-4 block flex items-center justify-center gap-2">
              <span className="material-symbols-outlined text-[16px]">groups</span>
              04 / Development Team
            </span>
            <h2 className="font-headline-md text-headline-md uppercase">Meet the Research Team</h2>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-gutter">
            
            {/* Author Card 1: Sayyam Akram */}
            <div className="border border-[rgba(255,255,255,0.08)] p-8 rounded-[20px] flex flex-col items-center text-center bg-[#1B1F24] hover:border-[#FF6B57] hover:-translate-y-[4px] hover:shadow-[0_10px_30px_rgba(0,0,0,0.25)] transition-all duration-[250ms] ease-out">
              <div className="w-16 h-16 border border-[rgba(255,255,255,0.08)] text-on-surface-variant rounded-full flex items-center justify-center mb-6">
                <span className="material-symbols-outlined text-[28px] text-on-surface-variant">person</span>
              </div>
              <h3 className="font-body-lg text-body-lg font-bold mb-1">Sayyam Akram</h3>
              <p className="font-label-sm text-label-sm text-on-surface-variant uppercase flex items-center gap-1">
                <span className="material-symbols-outlined text-[14px]">schema</span>
                Model Architecture
              </p>
            </div>

            {/* Author Card 2: Meer Hamza */}
            <div className="border border-[rgba(255,255,255,0.08)] p-8 rounded-[20px] flex flex-col items-center text-center bg-[#1B1F24] hover:border-[#FF6B57] hover:-translate-y-[4px] hover:shadow-[0_10px_30px_rgba(0,0,0,0.25)] transition-all duration-[250ms] ease-out">
              <div className="w-16 h-16 border border-[rgba(255,255,255,0.08)] text-on-surface-variant rounded-full flex items-center justify-center mb-6">
                <span className="material-symbols-outlined text-[28px] text-on-surface-variant">person</span>
              </div>
              <h3 className="font-body-lg text-body-lg font-bold mb-1">Meer Hamza</h3>
              <p className="font-label-sm text-label-sm text-on-surface-variant uppercase flex items-center gap-1">
                <span className="material-symbols-outlined text-[14px]">view_cozy</span>
                Web Pipeline
              </p>
            </div>

            {/* Author Card 3: Nuoman Yousaf */}
            <div className="border border-[rgba(255,255,255,0.08)] p-8 rounded-[20px] flex flex-col items-center text-center bg-[#1B1F24] hover:border-[#FF6B57] hover:-translate-y-[4px] hover:shadow-[0_10px_30px_rgba(0,0,0,0.25)] transition-all duration-[250ms] ease-out">
              <div className="w-16 h-16 border border-[rgba(255,255,255,0.08)] text-on-surface-variant rounded-full flex items-center justify-center mb-6">
                <span className="material-symbols-outlined text-[28px] text-on-surface-variant">person</span>
              </div>
              <h3 className="font-body-lg text-body-lg font-bold mb-1">Nuoman Yousaf</h3>
              <p className="font-label-sm text-label-sm text-on-surface-variant uppercase flex items-center gap-1">
                <span className="material-symbols-outlined text-[14px]">dataset</span>
                Data Processing
              </p>
            </div>

            {/* Author Card 4: Hassan Akram */}
            <div className="border border-[rgba(255,255,255,0.08)] p-8 rounded-[20px] flex flex-col items-center text-center bg-[#1B1F24] hover:border-[#FF6B57] hover:-translate-y-[4px] hover:shadow-[0_10px_30px_rgba(0,0,0,0.25)] transition-all duration-[250ms] ease-out">
              <div className="w-16 h-16 border border-[rgba(255,255,255,0.08)] text-on-surface-variant rounded-full flex items-center justify-center mb-6">
                <span className="material-symbols-outlined text-[28px] text-on-surface-variant">person</span>
              </div>
              <h3 className="font-body-lg text-body-lg font-bold mb-1">Hassan Akram</h3>
              <p className="font-label-sm text-label-sm text-on-surface-variant uppercase flex items-center gap-1">
                <span className="material-symbols-outlined text-[14px]">query_stats</span>
                Evaluation
              </p>
            </div>

          </div>
        </section>

        {/* Citation Section */}
        <section className="section-fade border-t border-[rgba(255,255,255,0.08)] pt-16" id="citation">
          <div className="bg-[#1B1F24] p-8 border border-[rgba(255,255,255,0.08)] rounded-[24px]">
            <div className="flex justify-between items-center mb-4">
              <h4 className="font-label-sm text-label-sm uppercase tracking-widest flex items-center gap-2">
                <span className="material-symbols-outlined text-[16px] text-primary">text_snippet</span>
                Cite BibTeX (arXiv 2023)
              </h4>
              <button 
                onClick={copyToClipboard}
                className="material-symbols-outlined text-primary hover:scale-110 transition-all flex items-center gap-1"
              >
                {copySuccess ? 'check_circle' : 'content_copy'}
              </button>
            </div>
            <pre className="font-code-md text-code-md text-on-surface-variant overflow-x-auto whitespace-pre-wrap">
{`@article{guo2023audio,
  title={Audio-Visual Instance Segmentation},
  author={Guo, Ruohao and Ying, Xianghua and Chen, Yaru and Niu, Dantong and Li, Guangyao and Qu, Liao and Qi, Yanyu and Zhou, Jinxing and Xing, Bowei and Yue, Wenzhen and Shi, Ji and Wang, Qixun and Zhang, Peiliang and Liang, Buwen},
  journal={arXiv preprint arXiv:2310.18709},
  year={2023}
}`}
            </pre>
          </div>
        </section>

      </main>

      {/* Footer */}
      <footer className="bg-surface-container-lowest border-t border-surface-container-highest">
        <div className="w-full py-16 px-margin-mobile md:px-margin-desktop max-w-max-width mx-auto flex flex-col md:flex-row justify-between gap-gutter">
          <div className="space-y-6">
            <div className="font-headline-md text-headline-md text-on-surface tracking-tight flex items-center gap-2">
              <span className="material-symbols-outlined text-primary">biotech</span>
              AVIS
            </div>
            <p className="font-label-sm text-label-sm text-on-surface-variant uppercase">GIFT University — Final Year Project</p>
            <div className="flex gap-4">
              <a className="bg-primary-container/10 border border-primary/20 px-4 py-2 text-primary font-label-sm uppercase tracking-widest hover:bg-primary-container hover:text-on-primary-container transition-all flex items-center gap-2" href="#">
                <span className="material-symbols-outlined text-[16px]">picture_as_pdf</span>
                Read AVIS Paper
              </a>
            </div>
            <p className="font-label-sm text-[10px] text-on-surface-variant/40">© 2024 Audio-Visual Instance Segmentation Research Lab.</p>
          </div>
          <div className="grid grid-cols-2 gap-x-12 gap-y-4">
            <a className="font-label-sm text-label-sm text-on-surface-variant hover:text-on-surface underline transition-all flex items-center gap-1" href="#citation">
              <span className="material-symbols-outlined text-[12px]">link</span>
              Cite BibTeX
            </a>
            <a className="font-label-sm text-label-sm text-on-surface-variant hover:text-on-surface underline transition-all flex items-center gap-1" href="#">
              <span className="material-symbols-outlined text-[12px]">code</span>
              GitHub
            </a>
            <a className="font-label-sm text-label-sm text-on-surface-variant hover:text-on-surface underline transition-all flex items-center gap-1" href="#">
              <span className="material-symbols-outlined text-[12px]">database</span>
              Dataset
            </a>
            <a className="font-label-sm text-label-sm text-on-surface-variant hover:text-on-surface underline transition-all flex items-center gap-1" href="#">
              <span className="material-symbols-outlined text-[12px]">school</span>
              University
            </a>
          </div>
        </div>
      </footer>
      </div>

      {/* Modern Split-screen Login / Sign Up Page */}
      {showAuthScreen && (
        <div 
          className={`fixed inset-0 z-[200] bg-[#08090a] flex items-center justify-center p-4 md:p-8 transition-all duration-[400ms] ease-out ${
            authTransitioning ? 'opacity-0 scale-[0.98] pointer-events-none' : 'opacity-100 scale-100'
          }`}
        >
          <div className="w-full max-w-[1200px] h-[760px] bg-[#111315] rounded-[30px] overflow-hidden border border-[rgba(255,255,255,0.08)] shadow-2xl flex flex-row">
            
            {/* Left half: Matte Charcoal Login/Signup Form */}
            <div className="w-full md:w-1/2 bg-[#111315] p-12 md:p-16 flex flex-col justify-between h-full select-none text-left">
              
              {/* Header */}
              <div className="space-y-3">
                <h2 className="text-[40px] font-bold text-white tracking-tight leading-none">Welcome</h2>
                <p className="text-[15px] text-[#b5bac4] font-semibold">
                  Create your account to <span className="text-[#FF6B57]">get started</span>
                </p>
              </div>

              {/* Segmented Pill Selector (Login / Sign Up) - Centered with margins */}
              <div className="my-6 relative flex bg-[#1a1d20] border border-[rgba(255,255,255,0.08)] rounded-full p-1.5 w-[240px]">
                <div 
                  className="absolute top-1.5 bottom-1.5 bg-[#FF6B57] rounded-full transition-all duration-300 ease-out"
                  style={{
                    width: 'calc(50% - 6px)',
                    left: authMode === 'signup' ? '6px' : 'calc(50%)'
                  }}
                />
                <button 
                  type="button" 
                  onClick={() => { setAuthMode('signup'); setAuthError(''); }}
                  className={`relative z-10 w-1/2 text-center text-[13px] font-bold uppercase transition-colors duration-200 py-2.5 ${authMode === 'signup' ? 'text-white' : 'text-[#b5bac4]'}`}
                >
                  Sign Up
                </button>
                <button 
                  type="button" 
                  onClick={() => { setAuthMode('login'); setAuthError(''); }}
                  className={`relative z-10 w-1/2 text-center text-[13px] font-bold uppercase transition-colors duration-200 py-2.5 ${authMode === 'login' ? 'text-white' : 'text-[#b5bac4]'}`}
                >
                  Login
                </button>
              </div>

              {/* Form with Horizontal slide, fade, and height transition */}
              <form onSubmit={handleAuthSubmit} className="space-y-6 my-auto relative">
                {authError && (
                  <div className="bg-red-500/10 border border-red-500/40 text-red-400 text-[13px] p-3 rounded-[16px] flex items-center gap-2">
                    <span className="material-symbols-outlined text-[16px]">error</span>
                    {authError}
                  </div>
                )}

                <div 
                  className="relative overflow-hidden w-full transition-all duration-300 ease-out"
                  style={{ 
                    height: authMode === 'signup' ? '236px' : '152px' 
                  }}
                >
                  <div 
                    className="absolute inset-0 w-full flex transition-all duration-300 ease-out"
                    style={{
                      transform: authMode === 'signup' ? 'translateX(0%)' : 'translateX(-100%)'
                    }}
                  >
                    {/* Sign Up Fields Container */}
                    <div className={`w-full shrink-0 pr-4 transition-opacity duration-300 ${authMode === 'signup' ? 'opacity-100' : 'opacity-0 pointer-events-none'} space-y-6`}>
                      <div className="relative flex items-center">
                        <span className="absolute left-5 text-[#5b636e] material-symbols-outlined">person</span>
                        <input
                          type="text"
                          required={authMode === 'signup'}
                          value={authName}
                          onChange={(e) => setAuthName(e.target.value)}
                          placeholder="Full Name"
                          className="w-full h-[60px] bg-[#1a1d20] text-white border border-[rgba(255,255,255,0.08)] rounded-[20px] pl-14 pr-6 text-[15px] font-medium placeholder-[#5b636e] focus:outline-none focus:border-[#FF6B57] focus:ring-0 transition-all duration-[250ms] ease-out"
                        />
                      </div>
                      <div className="relative flex items-center">
                        <span className="absolute left-5 text-[#5b636e] material-symbols-outlined">mail</span>
                        <input
                          type="email"
                          required={authMode === 'signup'}
                          value={authEmail}
                          onChange={(e) => setAuthEmail(e.target.value)}
                          placeholder="Email Address"
                          className="w-full h-[60px] bg-[#1a1d20] text-white border border-[rgba(255,255,255,0.08)] rounded-[20px] pl-14 pr-6 text-[15px] font-medium placeholder-[#5b636e] focus:outline-none focus:border-[#FF6B57] focus:ring-0 transition-all duration-[250ms] ease-out"
                        />
                      </div>
                      <div className="relative flex items-center">
                        <span className="absolute left-5 text-[#5b636e] material-symbols-outlined">lock</span>
                        <input
                          type={showPassword ? "text" : "password"}
                          required={authMode === 'signup'}
                          value={authPassword}
                          onChange={(e) => setAuthPassword(e.target.value)}
                          placeholder="Password"
                          className="w-full h-[60px] bg-[#1a1d20] text-white border border-[rgba(255,255,255,0.08)] rounded-[20px] pl-14 pr-12 text-[15px] font-medium placeholder-[#5b636e] focus:outline-none focus:border-[#FF6B57] focus:ring-0 transition-all duration-[250ms] ease-out"
                        />
                        <span 
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-5 text-[#5b636e] cursor-pointer hover:text-white transition-colors material-symbols-outlined select-none"
                        >
                          {showPassword ? "visibility_off" : "visibility"}
                        </span>
                      </div>
                    </div>

                    {/* Login Fields Container */}
                    <div className={`w-full shrink-0 pr-4 transition-opacity duration-300 ${authMode === 'login' ? 'opacity-100' : 'opacity-0 pointer-events-none'} space-y-6`}>
                      <div className="relative flex items-center">
                        <span className="absolute left-5 text-[#5b636e] material-symbols-outlined">mail</span>
                        <input
                          type="email"
                          required={authMode === 'login'}
                          value={authEmail}
                          onChange={(e) => setAuthEmail(e.target.value)}
                          placeholder="Email Address"
                          className="w-full h-[60px] bg-[#1a1d20] text-white border border-[rgba(255,255,255,0.08)] rounded-[20px] pl-14 pr-6 text-[15px] font-medium placeholder-[#5b636e] focus:outline-none focus:border-[#FF6B57] focus:ring-0 transition-all duration-[250ms] ease-out"
                        />
                      </div>
                      <div className="relative flex items-center">
                        <span className="absolute left-5 text-[#5b636e] material-symbols-outlined">lock</span>
                        <input
                          type={showPassword ? "text" : "password"}
                          required={authMode === 'login'}
                          value={authPassword}
                          onChange={(e) => setAuthPassword(e.target.value)}
                          placeholder="Password"
                          className="w-full h-[60px] bg-[#1a1d20] text-white border border-[rgba(255,255,255,0.08)] rounded-[20px] pl-14 pr-12 text-[15px] font-medium placeholder-[#5b636e] focus:outline-none focus:border-[#FF6B57] focus:ring-0 transition-all duration-[250ms] ease-out"
                        />
                        <span 
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-5 text-[#5b636e] cursor-pointer hover:text-white transition-colors material-symbols-outlined select-none"
                        >
                          {showPassword ? "visibility_off" : "visibility"}
                        </span>
                      </div>
                    </div>

                  </div>
                </div>

                <button
                  type="submit"
                  disabled={authSubmitting}
                  className="w-full h-[60px] bg-[#FF6B57] text-white hover:-translate-y-[2px] disabled:opacity-50 transition-all duration-[250ms] ease-out font-bold text-[15px] rounded-[20px] flex items-center justify-center gap-2 shadow-[0_4px_20px_rgba(255,107,87,0.15)]"
                >
                  {authSubmitting ? (
                    <>
                      <span className="material-symbols-outlined animate-spin text-[16px]">sync</span>
                      Processing...
                    </>
                  ) : (
                    authMode === 'signup' ? "Create Account" : "Login"
                  )}
                </button>

                <div className="relative flex items-center justify-center my-4">
                  <div className="border-t border-[rgba(255,255,255,0.08)] w-full"></div>
                  <span className="absolute bg-[#111315] px-3 text-[12px] text-[#5b636e] font-bold uppercase tracking-wider">or</span>
                </div>

                <button
                  type="button"
                  onClick={handleGoogleSignIn}
                  disabled={authSubmitting}
                  className="w-full h-[60px] bg-[#1a1d20] border border-[rgba(255,255,255,0.08)] text-white hover:bg-[#202428] hover:-translate-y-[2px] disabled:opacity-50 transition-all duration-[250ms] ease-out font-bold text-[15px] rounded-[20px] flex items-center justify-center gap-3 shadow-md"
                >
                  <svg className="w-5 h-5" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" />
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" />
                  </svg>
                  Continue with Google
                </button>
              </form>

              {/* Already have an account / Guest Mode */}
              <div className="space-y-6">
                <p className="text-[14px] text-center text-[#8b949e]">
                  {authMode === 'signup' ? "Already have an account? " : "Don't have an account? "}
                  <button
                    type="button"
                    onClick={() => { setAuthMode(authMode === 'signup' ? 'login' : 'signup'); setAuthError(''); }}
                    className="text-[#FF6B57] font-bold hover:underline"
                  >
                    {authMode === 'signup' ? 'Login' : 'Sign Up'}
                  </button>
                </p>

                {/* Guest Mode Trigger */}
                <div className="text-center pt-2">
                  <button
                    type="button"
                    onClick={() => {
                      const guest = { email: 'guest@avis.local', displayName: 'Guest User', uid: 'guest', mode: 'local' }
                      triggerTransition(guest)
                    }}
                    className="text-[12px] text-[#5b636e] uppercase tracking-wider hover:text-white transition-colors"
                  >
                    Continue as Guest
                  </button>
                </div>
              </div>

            </div>

            {/* Right half: Owl image visual split-mask */}
            <div className="w-1/2 relative hidden md:block overflow-hidden bg-[#0f1419] select-none">
              <img 
                src="/owl.png" 
                className="w-full h-full object-cover" 
                alt="AVIS Owl Segmentation" 
              />
              
              {/* Neon brand translucent overlay on left 50% */}
              <div className="absolute inset-y-0 left-0 w-1/2 bg-[#FF6B57]/25 pointer-events-none border-r border-[#FF6B57]/80"></div>
              
              {/* Vertical divider line & knob handle */}
              <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-[2px] bg-[#FF6B57]/90 flex items-center justify-center">
                <div className="w-6 h-10 bg-[#111315] border border-[rgba(255,255,255,0.08)] rounded-md flex flex-col justify-around items-center py-1.5 shadow-lg">
                  <div className="w-1 h-1 bg-white/60 rounded-full"></div>
                  <div className="w-1 h-1 bg-white/60 rounded-full"></div>
                  <div className="w-1 h-1 bg-white/60 rounded-full"></div>
                </div>
              </div>
            </div>

          </div>
        </div>
      )}

    </div>
  )
}

export default App

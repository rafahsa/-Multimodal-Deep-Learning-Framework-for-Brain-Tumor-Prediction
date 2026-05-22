import { Navbar, Footer, PageContainer } from './components/Layout';
import { BackendStatusBanner } from './components/common';
import { PredictPage } from './pages/PredictPage';
import { useBackendHealth } from './hooks/useBackendHealth';

function App() {
  const { state: backendState } = useBackendHealth();

  return (
    <>
      <Navbar
        links={[
          { label: 'Predict', href: '#' },
          { label: 'Docs', href: '/' },
        ]}
      />
      <BackendStatusBanner state={backendState} />
      <PageContainer>
        <PredictPage backendReady={backendState === 'healthy'} />
      </PageContainer>
      <Footer />
    </>
  );
}

export default App;

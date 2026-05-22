import { type CSSProperties, type ReactNode } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

const wrapper: CSSProperties = {
  maxWidth: '1180px',
  width: '100%',
  margin: '0 auto',
  padding: '6rem 2.5rem 3rem',
  flex: 1,
  background: ng.colors.bgPrimary,
};

interface PageContainerProps {
  children: ReactNode;
  style?: CSSProperties;
}

export function PageContainer({ children, style }: PageContainerProps) {
  return (
    <main className="page-container-inner" style={{ ...wrapper, ...style }}>
      {children}
    </main>
  );
}

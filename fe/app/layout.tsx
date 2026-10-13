import type { Metadata } from 'next';
import './globals.css';
import Nav from '@/components/Nav';

export const metadata: Metadata = {
  title: 'CCCD eKYC Test',
  description: 'Frontend test cho AI eKYC service (NLU/FE)',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body>
        <Nav />
        <main className="container">{children}</main>
      </body>
    </html>
  );
}

'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { api, clearToken, getToken, User } from '@/lib/api';

export default function Nav() {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    if (getToken()) {
      api.me().then(setUser).catch(() => {
        clearToken();
        setUser(null);
      });
    } else {
      setUser(null);
    }
  }, [pathname]);

  function logout() {
    clearToken();
    setUser(null);
    router.push('/login');
  }

  return (
    <nav className="nav">
      <Link href="/" className="brand">
        CCCD eKYC Test
      </Link>
      <div className="links">
        {user ? (
          <>
            <Link href="/kyc">KYC</Link>
            <Link href="/verifications">Lịch sử</Link>
            <span style={{ color: 'var(--muted)' }}>{user.email}</span>
            <button className="secondary" onClick={logout}>
              Đăng xuất
            </button>
          </>
        ) : (
          <>
            <Link href="/login">Đăng nhập</Link>
            <Link href="/register">Đăng ký</Link>
          </>
        )}
      </div>
    </nav>
  );
}

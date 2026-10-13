'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import Link from 'next/link';
import { api, setToken } from '@/lib/api';

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await api.register({ email, password, fullName });
      setToken(res.accessToken);
      router.push('/kyc');
    } catch (err: any) {
      setError(err.message || 'Đăng ký thất bại');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card" style={{ maxWidth: 420, margin: '40px auto 0' }}>
      <h2>Đăng ký</h2>
      {error && <div className="alert error">{error}</div>}
      <form onSubmit={onSubmit}>
        <label>Họ tên</label>
        <input type="text" value={fullName} onChange={(e) => setFullName(e.target.value)} />
        <label>Email</label>
        <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <label>Mật khẩu (tối thiểu 6 ký tự)</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          minLength={6}
          required
        />
        <button type="submit" disabled={loading} style={{ width: '100%' }}>
          {loading ? 'Đang tạo...' : 'Tạo tài khoản'}
        </button>
      </form>
      <p className="sub" style={{ marginTop: 16 }}>
        Đã có tài khoản? <Link href="/login">Đăng nhập</Link>
      </p>
    </div>
  );
}

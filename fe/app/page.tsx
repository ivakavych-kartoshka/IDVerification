import Link from 'next/link';

export default function Home() {
  return (
    <>
      <h1>CCCD eKYC Test Console</h1>
      <p className="sub">
        FE test gọi NestJS BE, BE gọi AI eKYC (FastAPI). Đăng nhập rồi vào KYC để upload CCCD +
        selfie.
      </p>

      <div className="card">
        <h2>Bắt đầu</h2>
        <div className="row">
          <Link href="/register">
            <button>Đăng ký</button>
          </Link>
          <Link href="/login">
            <button className="secondary">Đăng nhập</button>
          </Link>
          <Link href="/kyc">
            <button className="secondary">Vào KYC</button>
          </Link>
        </div>
      </div>

      <div className="card">
        <h2>Luồng</h2>
        <ol style={{ lineHeight: 1.9, color: 'var(--muted)', fontSize: 14 }}>
          <li>Upload CCCD mặt trước</li>
          <li>Upload CCCD mặt sau</li>
          <li>Chụp selfie bằng webcam (hoặc upload ảnh)</li>
          <li>Submit → BE tạo record + gọi AI service</li>
          <li>Hiển thị kết quả VERIFIED / REJECTED và chi tiết OCR / face / liveness</li>
        </ol>
      </div>
    </>
  );
}

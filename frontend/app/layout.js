import "./globals.css";

export const metadata = {
  title: "FinScout",
  description: "Annual-report QA demo with citations and PDF navigation",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

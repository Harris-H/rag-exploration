import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RAG 探索之旅",
  description: "从关键词搜索到向量检索，再到完整 RAG 管道的渐进式学习之旅",
};

const navLinks = [
  { href: "/", label: "首页", icon: "◈" },
  { href: "/route-a", label: "路线 A", icon: "A" },
  { href: "/route-b", label: "路线 B", icon: "B" },
  { href: "/route-c", label: "路线 C", icon: "C" },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="zh-CN"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-[#f8fafc] text-slate-900">
        {/* Top navigation */}
        <nav className="sticky top-0 z-50 bg-white border-b border-slate-200 shadow-sm backdrop-blur-xl">
          <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2 text-slate-900 hover:opacity-80 transition-opacity">
              <span className="text-lg">🔍</span>
              <span className="font-semibold text-sm tracking-tight">RAG 探索之旅</span>
            </Link>
            <div className="flex items-center gap-1">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="px-3 py-1.5 rounded-lg text-sm text-slate-600 hover:text-blue-600 hover:bg-slate-50 transition-all"
                >
                  <span className="mr-1.5 font-mono text-xs opacity-60">{link.icon}</span>
                  {link.label}
                </Link>
              ))}
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main className="flex-1">{children}</main>

        {/* Footer */}
        <footer className="border-t border-slate-200 py-6 text-center text-xs text-slate-400">
          RAG Exploration Demo · 渐进式学习检索增强生成
        </footer>
      </body>
    </html>
  );
}

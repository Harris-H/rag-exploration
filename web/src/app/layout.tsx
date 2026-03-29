import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
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
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
          {/* Top navigation */}
          <nav className="sticky top-0 z-50 bg-background/80 backdrop-blur-xl border-b border-border shadow-sm">
            <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
              <Link href="/" className="flex items-center gap-2 text-foreground hover:opacity-80 transition-opacity">
                <span className="text-lg">🔍</span>
                <span className="font-semibold text-sm tracking-tight">RAG 探索之旅</span>
              </Link>
              <div className="flex items-center gap-1">
                {navLinks.map((link) => (
                  <Link
                    key={link.href}
                    href={link.href}
                    className="px-3 py-1.5 rounded-lg text-sm text-muted-foreground hover:text-blue-600 dark:hover:text-blue-400 hover:bg-accent transition-all"
                  >
                    <span className="mr-1.5 font-mono text-xs opacity-60">{link.icon}</span>
                    {link.label}
                  </Link>
                ))}
                <ThemeToggle />
              </div>
            </div>
          </nav>

          {/* Main content */}
          <main className="flex-1">{children}</main>

          {/* Footer */}
          <footer className="border-t border-border py-6 text-center text-xs text-muted-foreground">
            RAG Exploration Demo · 渐进式学习检索增强生成
          </footer>
        </ThemeProvider>
      </body>
    </html>
  );
}

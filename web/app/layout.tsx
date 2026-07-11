import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const sans = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "SLM-125M · a legal language model trained from scratch",
  description:
    "A 125.8M-parameter Llama-style small language model pretrained from scratch on 2.04B tokens of US case law, SEC filings, and educational web text.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Set the saved theme before first paint to avoid a flash of the default.
  const themeScript = `(function(){try{var t=localStorage.getItem('slm-theme');if(t){document.documentElement.setAttribute('data-theme',t);}}catch(e){}})();`;

  return (
    <html lang="en" data-theme="parchment">
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className={`${sans.variable} ${mono.variable}`}>{children}</body>
    </html>
  );
}

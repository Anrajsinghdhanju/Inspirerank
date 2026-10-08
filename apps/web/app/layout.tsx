import "./globals.css";

export const metadata = {
  title: "InspireRank",
  description: "AI-powered personalized visual discovery",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

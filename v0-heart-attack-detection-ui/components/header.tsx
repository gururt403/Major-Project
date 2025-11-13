"use client"

import { Menu, Moon, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"

interface HeaderProps {
  onToggleDarkMode: () => void
  isDark: boolean
}

export default function Header({ onToggleDarkMode, isDark }: HeaderProps) {
  return (
    <header className="sticky top-0 z-50 bg-card border-b border-border shadow-sm">
      <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between md:px-6">
        {/* Logo and Project Name */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center text-primary-foreground font-bold text-lg">
            ❤️
          </div>
          <h1 className="text-xl font-bold text-foreground hidden sm:block">CardioWatch</h1>
        </div>

        {/* Navigation Links */}
        <nav className="hidden md:flex items-center gap-8">
          <a href="#dashboard" className="text-sm font-medium text-foreground hover:text-primary transition-colors">
            Dashboard
          </a>
          <a href="#monitoring" className="text-sm font-medium text-foreground hover:text-primary transition-colors">
            Live Monitoring
          </a>
          <a href="#reports" className="text-sm font-medium text-foreground hover:text-primary transition-colors">
            Reports
          </a>
          <a href="#about" className="text-sm font-medium text-foreground hover:text-primary transition-colors">
            About System
          </a>
        </nav>

        {/* Right Actions */}
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" onClick={onToggleDarkMode} className="rounded-lg">
            {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </Button>
          <Button variant="ghost" size="icon" className="md:hidden rounded-lg">
            <Menu className="w-5 h-5" />
          </Button>
        </div>
      </div>
    </header>
  )
}

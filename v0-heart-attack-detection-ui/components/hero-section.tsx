import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"

export default function HeroSection() {
  return (
    <section className="py-16 md:py-24 bg-gradient-to-b from-background to-muted/20">
      <div className="max-w-6xl mx-auto px-4 md:px-6 text-center">
        <div className="space-y-6">
          <div className="inline-block">
            <span className="px-3 py-1 bg-primary/10 text-primary text-xs font-semibold rounded-full">
              AI-Powered Health Monitoring
            </span>
          </div>

          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-foreground text-balance">
            Intelligent Heart Attack Detection System
          </h1>

          <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto text-balance">
            Real-time, contactless monitoring using advanced facial and posture analysis. Get instant alerts for
            potential heart stress patterns with our cutting-edge AI technology.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Button size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground gap-2 rounded-lg">
              Start Monitoring <ArrowRight className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="lg" className="rounded-lg bg-transparent">
              Learn More
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}

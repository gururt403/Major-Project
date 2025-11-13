import { Card, CardContent } from "@/components/ui/card"
import { CheckCircle } from "lucide-react"

const features = [
  {
    title: "Contactless AI",
    description: "Non-intrusive monitoring using advanced computer vision technology",
    icon: "📹",
  },
  {
    title: "rPPG Heart Rate",
    description: "Remote Photoplethysmography for accurate heart rate detection",
    icon: "❤️",
  },
  {
    title: "Posture Detection",
    description: "MHAVH-based posture analysis for comprehensive health assessment",
    icon: "🧍",
  },
  {
    title: "Real-time Alerts",
    description: "Instant notifications for abnormal patterns and potential risks",
    icon: "⚠️",
  },
]

export default function AboutSection() {
  return (
    <section id="about" className="py-16 md:py-24 bg-background">
      <div className="max-w-6xl mx-auto px-4 md:px-6">
        <div className="space-y-12">
          <div className="text-center space-y-4">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">About the System</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              CardioWatch represents cutting-edge AI technology designed to provide medical-grade heart health
              monitoring with user-friendly accessibility.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <Card key={index} className="border border-border hover:shadow-lg transition-all duration-300">
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    <div className="text-4xl">{feature.icon}</div>
                    <div>
                      <h3 className="font-semibold text-lg text-foreground flex items-center gap-2">
                        <CheckCircle className="w-5 h-5 text-primary" />
                        {feature.title}
                      </h3>
                      <p className="text-sm text-muted-foreground mt-2">{feature.description}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="border-2 border-primary/20 bg-primary/5">
            <CardContent className="pt-8">
              <h3 className="text-xl font-bold text-foreground mb-4">Technology Stack</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-semibold text-primary mb-2">Core Technologies</p>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Advanced Computer Vision</li>
                    <li>• Deep Learning Models</li>
                    <li>• Real-time Signal Processing</li>
                  </ul>
                </div>
                <div>
                  <p className="text-sm font-semibold text-primary mb-2">Algorithms</p>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• rPPG Analysis</li>
                    <li>• MHAVH Posture Model</li>
                    <li>• Anomaly Detection</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}

from manim import *
import random

class HappyNewYear2025(Scene):
    def construct(self):
        # Background color
        self.camera.background_color = DARK_BLUE

        # Title Text
        title = Text("Happy New Year 2025!", font_size=72, color=YELLOW).set_glow_factor(0.7)
        title.to_edge(UP)
        self.play(Write(title), run_time=3)

        # Add a glowing cross
        cross = Cross(color=GOLD).scale(2)
        cross.shift(DOWN * 2)
        self.play(Create(cross), run_time=2)

        # Add scripture text
        scripture = Text("'For I know the plans I have for you,' declares the Lord...\nJeremiah 29:11",
                         font_size=36, color=WHITE)
        scripture.next_to(cross, DOWN, buff=1.0)
        self.play(FadeIn(scripture), run_time=3)

        # Add fireworks
        for _ in range(5):
            x = random.uniform(-7, 7)
            y = random.uniform(-4, 4)
            firework = Star(color=random.choice([RED, ORANGE, PURPLE, GREEN, BLUE])).scale(0.5).move_to([x, y, 0])
            self.play(GrowFromCenter(firework), FadeOut(firework, run_time=2), run_time=2)

        # Add closing message
        closing_message = Text("Wishing You a Blessed and Joyful Year!", font_size=48, color=LIGHT_PINK)
        closing_message.to_edge(DOWN)
        self.play(FadeIn(closing_message), run_time=3)

        # Hold the scene for a moment
        self.wait(3)







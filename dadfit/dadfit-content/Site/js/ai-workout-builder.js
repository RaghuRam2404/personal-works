// AI Workout Builder - Generates ChatGPT Prompt with Exercise Library

// Exercise Database (same as workout-builder.js)
const exerciseLibrary = {
    core: {
        mcgillCurlUp: { name: "McGill Curl-Up", description: "Targets rectus abdominis and obliques for spinal stability", video: "https://youtu.be/2_e4I-brfqs", equipment: ["minimal", "home", "full"] },
        plank: { name: "Plank", description: "Core stabilization exercise", video: "https://www.youtube.com/shorts/v25dawSzRTM", equipment: ["minimal", "home", "full"] },
        shoulderTap: { name: "Shoulder Tap Plank", description: "Dynamic core stability with anti-rotation", video: "https://youtu.be/a2duDaO9BhM", equipment: ["minimal", "home", "full"] }
    },
    chest: {
        benchPress: { name: "Flat Barbell Bench Press", description: "Overall chest development, compound strength builder", video: "https://www.youtube.com/watch?v=4Y2ZdHCOXok", equipment: ["full"], compound: true },
        inclineDbPress: { name: "Incline Dumbbell Press", description: "Upper chest emphasis (clavicular head)", video: "https://youtu.be/0f6-uCUKqgA", equipment: ["home", "full"], compound: true },
        declinePress: { name: "Decline Bench Press", description: "Lower chest focus (costal head)", video: "https://youtu.be/LfyQBUKR8SE", equipment: ["full"], compound: true },
        chestFly: { name: "Dumbbell Chest Fly", description: "Pectoral isolation through horizontal adduction", video: "https://www.youtube.com/shorts/g3T7LsEeDWQ", equipment: ["home", "full"] },
        pushUp: { name: "Push-Ups", description: "Bodyweight chest, shoulders, and triceps builder", video: "", equipment: ["minimal", "home", "full"], compound: true }
    },
    back: {
        pullUp: { name: "Pull-Ups", description: "King of back exercises - lats, biceps, upper back", video: "https://www.youtube.com/watch?v=p40iUjf02j0", equipment: ["full"], compound: true },
        invertedRow: { name: "Inverted Rows (Bodyweight)", description: "Horizontal pull - back thickness, use table edge", video: "", equipment: ["minimal", "home", "full"], compound: true },
        latPulldown: { name: "Lat Pulldown (V-Grip)", description: "Lat emphasis with controlled resistance", video: "https://www.youtube.com/shorts/KV4D8MQrdhw", equipment: ["full"], compound: true },
        wideGripRow: { name: "Wide Grip Cable Row", description: "Middle/lower traps, rhomboids, rear delts for width", video: "https://www.youtube.com/shorts/KV4D8MQrdhw?t=20&feature=share", equipment: ["full"], compound: true },
        narrowGripRow: { name: "Narrow Grip Cable Row", description: "Lower lat emphasis with increased biceps involvement", video: "https://www.youtube.com/shorts/KV4D8MQrdhw?t=30&feature=share", equipment: ["full"], compound: true },
        facePull: { name: "Face Pull", description: "Posterior deltoids, trapezius, rhomboids for shoulder health", video: "https://www.youtube.com/shorts/8686PLZB_1Q", equipment: ["full"] },
        supermanHold: { name: "Superman Hold", description: "Lower back and posterior chain activation", video: "", equipment: ["minimal", "home", "full"] },
        deadlift: { name: "Conventional Deadlift", description: "Complete posterior chain: back, glutes, hamstrings", video: "https://youtu.be/hCDzSR6bW10", equipment: ["full"], compound: true }
    },
    shoulders: {
        pikePushup: { name: "Pike Push-Ups", description: "Bodyweight shoulder press alternative", video: "", equipment: ["minimal", "home", "full"], compound: true },
        overheadPress: { name: "Overhead Dumbbell Press", description: "Complete shoulder development with core stability", video: "https://www.youtube.com/shorts/OLePvpxQEGk", equipment: ["home", "full"], compound: true },
        lateralRaise: { name: "Dumbbell Lateral Raise", description: "Lateral deltoid isolation for shoulder width", video: "https://www.youtube.com/shorts/HeovYNoZDRg", equipment: ["home", "full"] },
        cableLateralRaise: { name: "Cable Lateral Raise", description: "Constant tension on lateral deltoids", video: "https://www.youtube.com/shorts/f_OGBg2KxgY", equipment: ["full"] },
        frontRaise: { name: "Front Raise", description: "Anterior deltoid targeting", video: "https://youtu.be/3sjY-whu6Uw", equipment: ["home", "full"] },
        rearDeltFly: { name: "Rear Delt Fly", description: "Posterior deltoid isolation", video: "https://youtu.be/P5CXx_jgTDE", equipment: ["home", "full"] },
        shrugs: { name: "Dumbbell Shrugs", description: "Trapezius development, scapular elevation", video: "https://www.youtube.com/shorts/YOGQaNTc470", equipment: ["home", "full"] }
    },
    biceps: {
        chinUp: { name: "Chin-Ups (Underhand)", description: "Bodyweight biceps and back builder", video: "", equipment: ["minimal", "home", "full"], compound: true },
        inclineCurl: { name: "Incline Dumbbell Curl", description: "Maximizes biceps stretch, emphasizes long head", video: "https://www.youtube.com/shorts/5hqKVbVn2iw", equipment: ["home", "full"] },
        hammerCurl: { name: "Hammer Curl", description: "Emphasizes brachioradialis and forearm muscles", video: "https://www.youtube.com/shorts/nFQI0c1AZgM", equipment: ["home", "full"] },
        crossBodyHammer: { name: "Cross-Body Hammer Curl", description: "Emphasizes brachialis for biceps thickness", video: "https://www.youtube.com/shorts/E0fhqFK3agg", equipment: ["home", "full"] },
        concentrationCurl: { name: "Concentration Curl", description: "Pure biceps isolation, eliminates momentum", video: "https://www.youtube.com/shorts/SfUrB2pkPGQ", equipment: ["home", "full"] }
    },
    triceps: {
        skullCrusher: { name: "Dumbbell Skull Crushers", description: "Full muscle stretch of long and lateral head", video: "https://youtu.be/8Nkfuhxsl-0?t=242", equipment: ["home", "full"] },
        cableKickback: { name: "Cable Triceps Kickback", description: "Peak contraction of long & lateral head", video: "https://youtu.be/8Nkfuhxsl-0?t=336", equipment: ["full"] },
        reverseGripPushdown: { name: "Reverse Grip Cable Pushdown", description: "Targets medial head (finish exercise)", video: "https://www.youtube.com/shorts/8IK6BkC0lWE?t=7&feature=share", equipment: ["full"] },
        tricepDips: { name: "Tricep Dips", description: "Bodyweight triceps builder (use chair or parallel bars)", video: "", equipment: ["minimal", "home", "full"], compound: true },
        diamondPushup: { name: "Diamond Push-Ups", description: "Triceps-focused push-up variation", video: "", equipment: ["minimal", "home", "full"] }
    },
    forearms: {
        wristCurls: { name: "Wrist Curls", description: "Forearm flexors, palm-up position, wrist flexion movement", video: "https://www.youtube.com/watch?v=3VLTzIrnb5g&pp=ygULd3Jpc3QgY3VybHM%3D", equipment: ["home", "full"] },
        reverseWristCurls: { name: "Reverse Wrist Curls", description: "Forearm extensors, palm-down position", video: "https://www.youtube.com/shorts/B699nq91i_w", equipment: ["home", "full"] },
        gripStrength: { name: "Grip Strength Training", description: "Farmer's carries, dead hangs, gripper work", video: "https://www.youtube.com/shorts/b_wVMXF7Hto", equipment: ["home", "full"] }
    },
    legs: {
        squats: { name: "Barbell Squats", description: "King of leg exercises - quads, glutes, hamstrings, core", video: "https://www.youtube.com/watch?v=my0tLDaWyDU", equipment: ["full"], compound: true },
        bodyweightSquat: { name: "Bodyweight Squats", description: "Fundamental squat pattern for beginners", video: "", equipment: ["minimal", "home", "full"], compound: true },
        jumpSquat: { name: "Jump Squats", description: "Explosive power and leg strength", video: "", equipment: ["minimal", "home", "full"], compound: true },
        hipThrust: { name: "Barbell Hip Thrust", description: "Superior glute activation and development", video: "https://youtu.be/SEdqd1n0cvg", equipment: ["home", "full"], compound: true },
        gluteBridge: { name: "Glute Bridges (Bodyweight)", description: "Glute activation and hip extension", video: "", equipment: ["minimal", "home", "full"], compound: true },
        romanianDeadlift: { name: "Romanian Deadlift", description: "Hamstrings, glutes, lower back - hip hinge mastery", video: "https://www.youtube.com/shorts/5rIqP63yWFg", equipment: ["home", "full"], compound: true },
        singleLegRDL: { name: "Single-Leg RDL (Bodyweight)", description: "Balance, hamstrings, glutes - unilateral strength", video: "", equipment: ["minimal", "home", "full"], compound: true },
        lunges: { name: "Walking Lunges", description: "Unilateral leg development and stability", video: "https://www.youtube.com/shorts/8JrHIIbdtTc", equipment: ["minimal", "home", "full"], compound: true },
        calfRaise: { name: "Calf Raises", description: "Seated: soleus | Standing: gastrocnemius", video: "https://www.youtube.com/shorts/baEXLy09Ncc", equipment: ["minimal", "home", "full"] }
    },
    abs: {
        crunches: { name: "Seated Crunches & Leg Raises", description: "Abdominal development variations", video: "https://youtu.be/Tn-XvYG9x7w?t=82", equipment: ["minimal", "home", "full"] }
    }
};

// Form Submission Handler
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('aiWorkoutForm');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Collect form data
        const preferences = {
            daysPerWeek: document.querySelector('input[name="daysPerWeek"]:checked').value,
            duration: document.querySelector('input[name="duration"]:checked').value,
            style: document.querySelector('input[name="style"]:checked').value,
            equipment: document.querySelector('input[name="equipment"]:checked').value,
            experience: document.querySelector('input[name="experience"]:checked').value,
            goal: document.querySelector('input[name="goal"]:checked').value
        };
        
        // Generate ChatGPT prompt
        const prompt = generateChatGPTPrompt(preferences);
        
        // Display the prompt
        displayPrompt(prompt);
        
        // Scroll to results
        document.getElementById('promptResult').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    });
});

// Generate ChatGPT Prompt
function generateChatGPTPrompt(preferences) {
    // Format preferences in readable text
    const goalText = {
        'muscle': 'Build Muscle (Hypertrophy)',
        'strength': 'Build Strength (Powerlifting style)',
        'fat-loss': 'Fat Loss (Higher reps, metabolic focus)',
        'general': 'General Fitness (Balanced approach)'
    }[preferences.goal];
    
    const styleText = {
        'normal': 'Normal Sets (Traditional rest between sets)',
        'superset': 'Supersets (Pair exercises back-to-back)',
        'circuit': 'Circuit Training (Fast-paced, minimal rest)'
    }[preferences.style];
    
    const equipmentText = {
        'full': 'Full Gym (Barbell, Dumbbells, Cables, Machines)',
        'home': 'Home Gym (Dumbbells only)',
        'minimal': 'Minimal Equipment (Bodyweight only)'
    }[preferences.equipment];
    
    const experienceText = {
        'beginner': 'Beginner (< 6 months training)',
        'intermediate': 'Intermediate (6 months - 2 years)',
        'advanced': 'Advanced (2+ years consistent training)'
    }[preferences.experience];
    
    // Format exercise library for the prompt
    let exerciseLibraryText = '';
    for (const [category, exercises] of Object.entries(exerciseLibrary)) {
        exerciseLibraryText += `\n${category.toUpperCase()}:\n`;
        for (const [key, exercise] of Object.entries(exercises)) {
            const compoundTag = exercise.compound ? ' [COMPOUND]' : '';
            const equipmentList = exercise.equipment.join(', ');
            exerciseLibraryText += `  • ${exercise.name}${compoundTag}\n`;
            exerciseLibraryText += `    Description: ${exercise.description}\n`;
            exerciseLibraryText += `    Equipment: ${equipmentList}\n`;
            exerciseLibraryText += `    Video: ${exercise.video}\n\n`;
        }
    }
    
    // Build the complete prompt
    const prompt = `I need you to create a personalized workout plan based on my preferences and the exercise library provided below.

MY WORKOUT PREFERENCES:
• Days Per Week: ${preferences.daysPerWeek} days
• Duration Per Session: ${preferences.duration} minutes
• Training Style: ${styleText}
• Equipment Available: ${equipmentText}
• Experience Level: ${experienceText}
• Primary Goal: ${goalText}

INSTRUCTIONS:
1. Create a ${preferences.daysPerWeek}-day workout split appropriate for my experience level and goals
2. Each workout should fit within ${preferences.duration} minutes including all phases
3. Use ONLY exercises from the library below that match my equipment (${preferences.equipment})
4. Prioritize compound movements (marked [COMPOUND]) for efficiency
5. Adjust sets/reps based on my goal:
   - Muscle Building: 3-4 sets × 6-12 reps
   - Strength: 3-5 sets × 4-6 reps
   - Fat Loss: 3 sets × 12-15 reps (IMPORTANT: Keep rest periods SHORT - 30-45 seconds max for metabolic conditioning)
   - General Fitness: 3 sets × 8-12 reps
6. For ${preferences.style === 'superset' ? 'Supersets: Pair complementary exercises' : preferences.style === 'circuit' ? 'Circuit Training: List exercises to be performed in sequence with minimal rest' : 'Normal Sets: Allow 2-3 minutes rest between compound exercises'}${preferences.goal === 'fat-loss' ? '\n   SPECIAL NOTE FOR FAT LOSS: Keep rest periods minimal (30-45 seconds) to maintain elevated heart rate and maximize calorie burn' : ''}
7. CRITICAL: Follow the OPTIMAL WORKOUT SEQUENCE below for each workout
8. Include the video link for each exercise

OPTIMAL WORKOUT SEQUENCE (Science-Based 5-Step Framework):
⚠️ THIS SEQUENCE IS CRITICAL FOR INJURY PREVENTION AND MAXIMUM RESULTS

STEP 1: DYNAMIC WARM-UP (5-10 minutes)
Purpose: Movement prep, elevate heart rate, increase muscle temperature
- Light cardio: Jumping jacks, high knees, or brisk walking (2-3 minutes)
- Dynamic stretches for neck and upper/lower body: Arm circles, leg swings, torso twists, hip circles (3-5 minutes)
- Movement prep relevant to workout focus (2-3 minutes)

STEP 2: CORE ACTIVATION & STABILISATION (3-5 minutes)
Purpose: Engage stabilizers before loading spine, prevent injury
- McGill Big 3 exercises: McGill Curl-Up, Plank, Shoulder Tap
- For UPPER BODY DAYS ONLY: Add rotator cuff exercises (side-lying external rotation, banded W, full can hold)
Note: This step is CRITICAL for spinal stability during heavy lifts

STEP 3: MAIN WORKOUT (${Math.max(preferences.duration - 25, 20)} minutes)
Purpose: Primary strength/hypertrophy work
⚠️ EXERCISE ORDER MATTERS:
   1. Compound lifts FIRST (marked [COMPOUND]) - train in order of movement complexity
   2. Isolation exercises SECOND - target specific muscles after compounds
Reasoning: Compounds require most energy and technique, do when fresh

STEP 4: CARDIO FINISHER (Optional, 5-10 minutes)
Purpose: Cardiovascular conditioning without compromising strength gains
- HIIT option: 20 seconds work / 40 seconds rest × 6-8 rounds
- Steady-state option: Light jog, bike, or rowing
Note: Skip or shorten if time-constrained or prioritizing pure strength

STEP 5: STATIC STRETCHING (5 minutes)
Purpose: Cool-down phase, improve flexibility while muscles warm, aid recovery
- Hold each stretch 20-30 seconds:
  * Hamstring stretch, Quad stretch, Hip flexor stretch
  * Chest/shoulder stretch, Back/lat stretch
  * Focus on muscles worked during this session

EXERCISE LIBRARY:
${exerciseLibraryText}

WORKOUT STRUCTURE TEMPLATE:
Each workout day should follow this format:

═══════════════════════════════════════
💬 Hey, quick note from Coach Raghu
═══════════════════════════════════════

This AI workout is a solid starting point! But if you ever feel stuck with form, nutrition, or just need someone to bounce ideas off - I'm here.

I work with Indian dads who are juggling family, work, and fitness. No cookie-cutter plans, just real conversations about what works for YOUR life.

If you'd like to chat: https://dadfit.in/contact

(We can talk about workout tweaks, nutrition without crash diets, or just accountability check-ins over WhatsApp. Whatever helps!)

═══════════════════════════════════════

OUTPUT FORMAT:
Please structure the workout plan as:

═══════════════════════════════════════
DAY 1: [Focus Area - e.g., Upper Body Push]
═══════════════════════════════════════

STEP 1️⃣: DYNAMIC WARM-UP (5-10 min)
- [Specific dynamic stretches for today's muscle groups]
- [Movement prep relevant to exercises]

STEP 2️⃣: CORE ACTIVATION & STABILISATION (3-5 min)
- McGill Curl-Up: [sets] × [reps/time]
  Video: https://youtu.be/2_e4I-brfqs
- Plank: [sets] × [time]
  Video: https://www.youtube.com/shorts/v25dawSzRTM
${preferences.equipment !== 'minimal' ? '- [Add rotator cuff if upper body day]' : ''}

STEP 3️⃣: MAIN WORKOUT (${Math.max(preferences.duration - 25, 20)} min)

COMPOUND EXERCISES (Do FIRST):
1. [Exercise Name] - [Sets] × [Reps]
   Description: [Brief form cue]
   Video: [link]
   Rest: [time between sets]
   
2. [Exercise Name] - [Sets] × [Reps]
   Description: [Brief form cue]
   Video: [link]
   Rest: [time between sets]

ISOLATION EXERCISES (Do SECOND):
3. [Exercise Name] - [Sets] × [Reps]
   Description: [Brief form cue]
   Video: [link]
   Rest: [time between sets]
   
(Continue for all exercises)

STEP 4️⃣: CARDIO FINISHER (Optional, 5-10 min)
- HIIT Option: [Specific cardio exercise] 20 sec work / 40 sec rest × 6-8 rounds
- OR Steady-State: Light jog/bike/rowing at conversational pace

STEP 5️⃣: STATIC STRETCHING (5 min)
- [Specific stretches for muscles worked today, hold 20-30 seconds each]

---

DAY 2: [Focus Area]
...

(Continue for all ${preferences.daysPerWeek} days)

═══════════════════════════════════════
ADDITIONAL GUIDELINES:
═══════════════════════════════════════

📋 Split Structure Explanation:
[Explain why this split works for the user's goals]

⏱️ Rest Periods:${preferences.goal === 'fat-loss' ? '\n⚠️ FAT LOSS FOCUS: Keep rest periods SHORT for maximum calorie burn\n- Compound exercises: 45-60 seconds\n- Isolation exercises: 30-45 seconds\n- Between circuits/supersets: 30 seconds' : '\n- Compound exercises: [X] minutes\n- Isolation exercises: [Y] seconds\n- Between circuits/supersets: [Z] seconds'}

💡 Pro Tips:
- [Important form cues]
- [Progressive overload recommendations]
- [Common mistakes to avoid]

🎯 Progress Tracking:
- [How to track and progress over time]

Generate the complete workout plan now with all warmups, main workouts, and cooldowns.`;

    return prompt;
}

// Display Prompt
function displayPrompt(prompt) {
    const resultSection = document.getElementById('promptResult');
    const promptOutput = document.getElementById('promptOutput');
    
    promptOutput.textContent = prompt;
    resultSection.style.display = 'block';
}

// Copy to Clipboard
function copyPrompt() {
    const promptText = document.getElementById('promptOutput').textContent;
    const copyBtn = document.getElementById('copyBtn');
    
    navigator.clipboard.writeText(promptText).then(function() {
        // Update button text
        copyBtn.innerHTML = '✅ Copied!';
        copyBtn.classList.add('copied');
        
        // Reset after 2 seconds
        setTimeout(function() {
            copyBtn.innerHTML = '📋 Copy to Clipboard';
            copyBtn.classList.remove('copied');
        }, 2000);
    }).catch(function(err) {
        console.error('Failed to copy: ', err);
        alert('Failed to copy. Please select and copy manually.');
    });
}

// Open ChatGPT with prompt
function openChatGPT() {
    const promptText = document.getElementById('promptOutput').textContent;
    
    // Copy to clipboard
    navigator.clipboard.writeText(promptText).then(function() {
        // Open ChatGPT in new tab
        window.open('https://chatgpt.com/', '_blank');
        
        // Show instruction to paste
        alert('✅ Prompt copied to clipboard!\n\n👉 ChatGPT is opening in a new tab.\n\nSimply PASTE the prompt (Ctrl+V or Cmd+V) into the chat to get your personalized workout plan.');
    }).catch(function(err) {
        console.error('Failed to copy: ', err);
        alert('❌ Failed to copy automatically.\n\nPlease manually copy the prompt from the box below and paste it in ChatGPT.');
        window.open('https://chatgpt.com/', '_blank');
    });
}

// Open Claude with prompt
function openClaude() {
    const promptText = document.getElementById('promptOutput').textContent;
    
    // Copy to clipboard
    navigator.clipboard.writeText(promptText).then(function() {
        // Open Claude in new tab
        window.open('https://claude.ai/', '_blank');
        
        // Show instruction to paste
        alert('✅ Prompt copied to clipboard!\n\n👉 Claude is opening in a new tab.\n\nSimply PASTE the prompt (Ctrl+V or Cmd+V) into the chat to get your personalized workout plan.');
    }).catch(function(err) {
        console.error('Failed to copy: ', err);
        alert('❌ Failed to copy automatically.\n\nPlease manually copy the prompt from the box below and paste it in Claude.');
        window.open('https://claude.ai/', '_blank');
    });
}

// Open Gemini with prompt
function openGemini() {
    const promptText = document.getElementById('promptOutput').textContent;
    
    // Copy to clipboard
    navigator.clipboard.writeText(promptText).then(function() {
        // Open Gemini in new tab
        window.open('https://gemini.google.com/', '_blank');
        
        // Show instruction to paste
        alert('✅ Prompt copied to clipboard!\n\n👉 Gemini is opening in a new tab.\n\nSimply PASTE the prompt (Ctrl+V or Cmd+V) into the chat to get your personalized workout plan.');
    }).catch(function(err) {
        console.error('Failed to copy: ', err);
        alert('❌ Failed to copy automatically.\n\nPlease manually copy the prompt from the box below and paste it in Gemini.');
        window.open('https://gemini.google.com/', '_blank');
    });
}

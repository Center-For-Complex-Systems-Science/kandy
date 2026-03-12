#!/bin/bash
# KANDy project splash screen for Claude Code SessionStart hook

# ── Colors ──────────────────────────────────────────────────────
C=$'\033[0;36m'     # cyan
BC=$'\033[1;36m'    # bold cyan
Y=$'\033[1;33m'     # bold yellow
G=$'\033[0;32m'     # green
M=$'\033[0;35m'     # magenta
BM=$'\033[1;35m'    # bold magenta
W=$'\033[1;37m'     # bold white
D=$'\033[2m'        # dim
R=$'\033[0m'        # reset

cat <<EOF

${D}┌──────────────────────────────────────────────────────────────────────┐${R}
${D}│${R}                                                                      ${D}│${R}
${D}│${R}   ${BC}██╗  ██╗ █████╗ ███╗   ██╗██████╗ ${R}                              ${D}│${R}
${D}│${R}   ${BC}██║ ██╔╝██╔══██╗████╗  ██║██╔══██╗${R}                              ${D}│${R}
${D}│${R}   ${BC}█████╔╝ ███████║██╔██╗ ██║██║  ██║${R} ${C}██╗   ██╗${R}                   ${D}│${R}
${D}│${R}   ${BC}██╔═██╗ ██╔══██║██║╚██╗██║██║  ██║${R} ${C}╚██╗ ██╔╝${R}                   ${D}│${R}
${D}│${R}   ${BC}██║  ██╗██║  ██║██║ ╚████║██████╔╝${R}  ${C}╚████╔╝${R}                    ${D}│${R}
${D}│${R}   ${BC}╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝${R}   ${C}╚██╔╝${R}                     ${D}│${R}
${D}│${R}                                          ${C}██╔╝${R}                      ${D}│${R}
${D}│${R}                                          ${C}╚═╝${R}                       ${D}│${R}
${D}│${R}                                                                      ${D}│${R}
${D}│${R}   ${W}⚡ Kolmogorov–Arnold Networks for Dynamics${R}                       ${D}│${R}
${D}│${R}   ${D}📐 ẋ = A · Ψ(φ(x))${R}                                               ${D}│${R}
${D}│${R}                                                                      ${D}│${R}
${D}├─────────────────────────┬────────────────────────────────────────────┤${R}
${D}│${R}                         ${D}│${R}                                            ${D}│${R}
${D}│${R}  ${Y}🔬 SKILLS${R}               ${D}│${R}  ${BM}🤖 AGENTS${R}                                ${D}│${R}
${D}│${R}  ${D}─────────${R}               ${D}│${R}  ${D}────────${R}                                  ${D}│${R}
${D}│${R}  ${G}/run-experiment${R} <sys>   ${D}│${R}  ${M}🧬 kandy-researcher${R}                      ${D}│${R}
${D}│${R}  ${G}/run-baselines${R}  <name>  ${D}│${R}  ${D}   Discover governing equations${R}           ${D}│${R}
${D}│${R}  ${G}/assess${R}         <sys>   ${D}│${R}  ${M}📊 model-quality-assessor${R}                 ${D}│${R}
${D}│${R}  ${G}/report${R}         <sys>   ${D}│${R}  ${D}   Evaluate model predictions${R}             ${D}│${R}
${D}│${R}  ${G}/list-systems${R}           ${D}│${R}  ${M}🧫 data-fitter${R} ${D}⌛ coming soon${R}          ${D}│${R}
${D}│${R}                         ${D}│${R}  ${D}   Fit models to real-world data${R}         ${D}│${R}
${D}│${R}  ${Y}🧪 CLI${R}                  ${D}│${R}  ${D}   iEEG · PDEs · ODEs · maps${R}             ${D}│${R}
${D}│${R}  ${D}────${R}                    ${D}│${R}                                            ${D}│${R}
${D}│${R}  ${G}uv run kandy${R} <sys>      ${D}│${R}                                            ${D}│${R}
${D}│${R}  ${G}uv run kandy${R} --list     ${D}│${R}                                            ${D}│${R}
${D}│${R}                         ${D}│${R}                                            ${D}│${R}
${D}└─────────────────────────┴────────────────────────────────────────────┘${R}

EOF

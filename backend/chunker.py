from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def create_optimal_splitter(chunk_size: int = 300, overlap: int = 80) -> RecursiveCharacterTextSplitter:
    """创建优化的分割器，按自然边界分块"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " "],
        length_function=len
    )

def chunk_document(text: str, metadata: dict = None) -> list[dict]:
    """分块并保留原数据"""
    splitter = create_optimal_splitter()
    chunks = splitter.split_text(text)
    return [
        {"content": chunk, "metadata": metadata or {}, "index": i} for i, chunk in enumerate(chunks)
    ]


async def main():
    doc = """[万字长文预警] Harness 最佳实践：在 Java Spring Boot 项目中落地 OpenSpec + Claude Code
        这篇文章不是在讲“怎么让 AI 多写一点代码”，而是在讲：怎么把 AI 放进一套可控、可审计、可复用的工程流程里。

        对于一个真实的生产级业务项目来说，最危险的从来不是“AI 写不出来”，而是：

        它在没有边界的情况下乱改；
        它不知道项目里的历史隐性约定；
        它把实现、评审、验证混在一起；
        它改完了代码，却没有把风险说清楚；
        它碰到了 SQL、配置、脚本这类高风险区域，却没有硬护栏。
        因此，harness 的目标不是“让 AI 更自由”，而是：

        让需求先变成工件，让代码只在护栏里改，让评审和验证分开，让仓库本身沉淀为知识库。

        这篇文章基于一次真实的项目实践，整理出一套适合 Java Spring Boot 项目的 harness 最佳实践。当然，harness 思想本身和语言是没有关系的，弄懂思想其它语言也一样。

        一、先说结论：这套 harness 的核心原则
        如果只保留最关键的几条原则，我会总结为下面 4 条：

        OpenSpec 管变更生命周期
        用 /opsx:propose -> /opsx:apply -> /opsx:verify -> /opsx:archive 管住“需求从提出到归档”的全过程。

        AGENTS.md 只当地图，不当百科全书
        AGENTS.md 负责告诉 AI“先看什么、按什么流程做”；真正的项目知识应该放在 docs/ 里。

        Claude 的硬约束靠 permissions + hooks
        规则写在文档里只是“软约束”；真正能拦住危险动作的，是权限和 hook。

        团队专用动作放到 skills 和 subagents
        比如 review 摘要、Spring 分层审查、SQL 风险审查，这些都应该变成可重复调用的能力，而不是每次靠人临时提醒。

        如果要再压缩成一句话，那就是：

        需求先工件化，知识先显性化，执行先加护栏，评审与验证必须分离。

        只有这样，AI 才不是一个“会写代码但不受控的助手”，而是一套真正可以进入生产研发流程的工程能力。

        Gemini_Generated_Image_8xqyem8xqyem8xqy 1.png
        二、这套方案适合什么项目
        这套 harness 最适合下面这类项目：

        有一定历史包袱的 Java Spring Boot 业务系统
        需求经常是“增量改造”，不是全新重写
        前后端存在一些口头约定或隐性契约
        团队希望把 AI 编码变成“流程能力”，而不是个人技巧
        需要把“需求、实现、评审、校验”明确拆开
        如果只是一个很小的 demo 仓库、一次性脚本仓库、纯实验项目，甚至就是一个从0到1的新项目，或许也可以考虑下和其它框架进行搭配，比如Superpowers、ecc、omc等等。现在这套配置会在约束层偏重，最好是根据具体的项目来进行自己的搭配，不一定需要完全照搬。

        三、推荐的仓库组织方式
        一个适合 harness 落地的 Java Spring Boot 仓库，建议像下面这样组织：

        repo/
        ├─ AGENTS.md                             # ├─ 🌟 通用OpenSpec规则
        ├─ CLAUDE.md                             # ├─ 🌟 Claude系统提示词
        ├─ REVIEW.md                             # ├─ 🌟 只读评审代理提示词
        ├─ docs/                                 # ├─ 📚 项目知识库目录
        │  ├─ architecture/                      # │  ├─ 🔥 整体知识
        │  │  └─ index.md                        # │  │  └─ 项目架构总览
        │  │  └─ implicit-contracts.md           # │  │  └─ 隐性业务约定/项目坑点
        │  ├─ product/                           # │  ├─ 🔥 产品知识
        │  │  └─ index.md                        # │  │  └─ 产品规则
        │  ├─ standards/                         # │  ├─ 🔥 相关规范
        │  │  ├─ testing.md                      # │  │  ├─ 测试规范
        │  │  └─ database.md                     # │  │  └─ 数据库与 SQL 规范
        ├─ openspec/                             # ├─ 📚 OpenSpec执行目录
        │  ├─ changes/                           # │  ├─ 变更目录
        │  │  ├─ <changes名>                     # │  ├─ 当前正在执行的change
        │  │  │  ├─ specs/                       # │  │  │  ├─ 该change的是怎么工作的
        │  │  │  ├─ proposal.md                  # │  │  │  ├─ 当前拆解的需求实现提案
        │  │  │  ├─ design.md                    # │  │  │  ├─ 执行的具体方案
        │  │  │  └─ task.md                      # │  │  │  └─ 拆解的执行步骤节点
        │  │  └─ archive/                        # │  │  └─ 归档文件
        │  └─ specs/                             # │  └─ 当前系统如何工作的
        ├─ .claude/                              # ├─ 📚 Claude项目级配置目录
        │  ├─ settings.local.json.example        # │  ├─ 🔥 项目级权限设置      
        │  ├─ skills/                            # │  ├─ 🔥 项目级Skills
        │  │  ├─ prepare-review/                 # │  │  ├─ review前变更审计
        │  │  │  └─ SKILL.md                     # │  │  │
        │  │  ├─ spring-architecture-review/     # │  │  ├─ Springboot分层架构检查
        │  │  │  └─ SKILL.md                     # │  │  │
        │  │  └─ sql-risk-review/                # │  │  └─ SQL、Mapper、批量更新等检查
        │  │     └─ SKILL.md                     # │  │  │
        │  ├─ agents/                            # │  ├─ 🔥 子代理
        │  │  └─ reviewer.md                     # │  │  └─ 只读评审代理
        │  └─ hooks/                             # │  └─ 🔥 Hook
        │     ├─ guard_write.py                  # │     ├─ 文件写入保护
        │     ├─ ensure_change_context.py        # │     ├─ 上下文变更保护
        │     └─ run_checks.sh                   # │     └─ 编译检查
        ├─ src/                                            
        │  ├─ main/
        │  │  ├─ java/
        │  │  └─ resources/
        │  └─ test/
        │     ├─ java/
        │     └─ resources/
        ├─ pom.xml
        └─ .gitignore
        这套结构里，最重要的不是“文件多”，而是职责分层清楚：

        openspec/：OpenSpec的执行目录，管理“这次改什么”
        docs/：你们公司或者项目对应的知识库，管理“这个项目本来是怎么工作的”
        AGENTS.md / CLAUDE.md / REVIEW.md：通用项目级配置，管理“AI 进入仓库后应该怎么做”
        .claude/settings.json + hooks：约束和限制配置，管理“哪些事不能做、哪些检查必须跑”
        skills / agents：自动化流程，管理“团队专用的审查动作”
        四、最佳实践之一：AGENTS.md 只做导航，不做知识库
        这是整个 harness 里非常关键的一点。

        很多人会本能地把所有规则都塞进 AGENTS.md，但这会很快失控。
        更好的方式是：

        AGENTS.md 只负责告诉 AI 去哪里看
        真正的知识、规则、隐性约定放进 docs/
        也就是说，AGENTS.md 应该像一个“入口地图”，而不是一本“项目百科全书”。

        它至少应该说明：

        仓库采用什么工作流
        AI 进来后必须先读哪些文件
        没有 change 是否允许开始开发
        哪些目录是受保护的
        主要命令入口是什么
        如果 AGENTS.md 变得越来越长，通常不是因为规则更多了，而是因为知识没有被正确拆分到 docs。

        五、最佳实践之二：把“隐性约定”单独文档化
        对真实业务项目来说，最危险的通常不是显式规则，而是那些“大家都知道，但没人写下来”的东西。

        例如这次实践里，至少就存在两类典型的隐性约定：

        status = null 和 status = 0 在历史语义上并不等价
        单元详情接口里，前端依赖 contentResponse 做回显
        这些内容如果只存在于口头沟通里，AI 基本不可能稳定推断出来。
        所以最佳实践是：

        把所有容易导致“改完能跑、联调出错”的口头约定，单独沉淀到 docs/architecture/implicit-contracts.md。

        并且要做到两件事：

        OpenSpec 在 design.md 阶段就显式检查这些约定
        reviewer / verify 阶段也把这些约定作为对齐依据
        这一步做得好不好，几乎直接决定 AI 产出的可落地程度。

        六、最佳实践之三：OpenSpec 只管“变更生命周期”，不要拿它代替全部治理
        OpenSpec 的定位要非常清楚：

        它负责把需求变成 change 工件，并驱动 change 生命周期。

        在 Spring Boot 项目里，我建议用下面这条主流程：

        /opsx:propose -> /opsx:apply -> /opsx:verify -> /opsx:archive
        其中每一步的职责应该严格区分：

        1. /opsx:propose
        把需求拆成一个 change，产出：

        proposal.md
        design.md
        tasks.md
        这里要强调一个实战经验：

        第一版 proposal 往往不靠谱，不要急着执行。

        它更像第一轮工作草案，而不是最终方案。
        如果 proposal 拆错了，宁可废弃当前 change，重新生成新的 change，也不要硬着头皮继续做。

        2. /opsx:apply
        根据已经确认过的 design.md 和 tasks.md 实施代码改动。
        这一步的重点不是“让 AI 尽量多做”，而是：

        只做 tasks.md 范围内的事
        不允许自行扩需求
        每完成一个里程碑就跑检查
        3. /opsx:verify
        核对“实现有没有和 OpenSpec 工件对上”。

        注意，verify 的作用非常重要，但也非常有限：

        /opsx:verify 不是代码评审，也不是架构评审，它只负责检查实现与 change 工件是否一致。

        4. /opsx:archive
        把当前 change 归档，保持 openspec/changes/ 的进行中上下文干净，方便后续进入下一个 change。

        七、最佳实践之四：把“实现、评审、验证”彻底拆开
        这是 harness 和普通 AI 编码方式最大的区别之一。

        在真实项目里，下面几件事不能混在一起：

        需求拆解
        代码实现
        OpenSpec 对齐校验
        架构审查
        SQL 风险审查
        面向 PR 的人工阅读摘要
        最佳实践是把它们拆成不同职责：

        /opsx:verify：检查“实现有没有和 OpenSpec 对上”
        /prepare-review：整理“这次到底改了什么，方便人 review”
        /spring-architecture-review：检查 Spring 分层有没有乱
        /sql-risk-review：检查 SQL、Mapper、批量更新、索引和扫描范围风险
        reviewer 子代理：做一轮独立、只读、偏代码审查视角的检查
        这样拆开的好处有两个：

        每种审查都有明确目标，不会互相覆盖又互相遗漏
        出问题时更容易定位，到底是 proposal 有问题、实现有问题，还是架构/SQL 有风险
        八、最佳实践之五：真正的硬护栏必须落在 permissions + hooks
        很多团队喜欢把规则写进提示词，然后默认 AI 会听。
        但在工程实践里，这种做法并不可靠。

        最佳实践是：

        能通过权限系统和 hook 强制拦住的，就不要只靠提示词约束。

        例如下面这些目录，就应该默认视为高风险区域：

        src/main/resources/application*.yml
        src/main/resources/bootstrap*.yml
        src/main/resources/db/
        sql/
        deploy/
        infra/
        secrets/
        对于这些路径，建议直接：

        在 permissions.deny 中禁止修改
        在 guard_write.py 里再做一层路径校验
        这样做的价值是：
        即使模型一时判断失误，也不会直接碰到最危险的区域。

        同理，对于 Bash 执行也应该分层控制：

        mvn test
        mvn -q -DskipTests compile
        mvn -DskipTests package
        git status
        git diff
        这类安全命令可以放入 allow。

        但像下面这些就应该默认禁止：

        git push
        kubectl
        terraform
        helm
        rm -rf
        任何生产部署相关命令
        九、最佳实践之六：Hooks 不只做拦截，还要做自动检查
        在 Claude Code 场景下，hooks 最有价值的地方，不只是阻止危险动作，还包括自动补上执行后校验。例如：

        1. 写入前校验
        通过 guard_write.py 拦截受保护路径写入。

        2. 命令前检查上下文
        通过 ensure_change_context.py 检查当前是否存在 OpenSpec change。
        没有 change 时，对高风险 Bash 动作直接 ask，而不是默认放行。

        3. 写入后自动跑检查
        通过 run_checks.sh 在代码变更后自动执行：

        编译检查
        单元测试
        打包检查
        同时，像文档类变更可以跳过重检查，避免无谓消耗。

        这套设计的关键点是：

        让“检查会不会跑”不再依赖模型自觉，而变成流程自动发生。

        十、最佳实践之七：Skill 和 Subagent 只做团队专用能力
        很多能力不适合硬塞进主提示词里，比如：

        生成 PR 前的 review 摘要
        检查 Spring Boot 分层架构
        检查 SQL 风险
        做一轮只读视角的 reviewer 审计
        这些能力最适合沉淀成：

        .claude/skills/*
        .claude/agents/*
        这样做有三个好处：

        可复用：以后每个 change 都可以重复调用
        可组合：不同类型的需求可以只调用需要的能力
        可演进：团队后续可以持续补充自己的审查能力，而不必频繁改系统提示词
        换句话说：

        主流程负责通用约束，skills / agents 负责团队私有经验。

        十一、标准工作流建议
        结合这次实践，我更推荐下面这套标准工作流：

        第 0 步：初始化仓库
        通过 openspec init --tools claude 生成基础目录，再补齐：

        AGENTS.md
        CLAUDE.md
        REVIEW.md
        docs/
        .claude/settings.json
        .claude/hooks/
        .claude/skills/
        .claude/agents/
        第 1 步：创建 change
        执行 /opsx:propose，让需求先变成工件。

        第 2 步：人工审 proposal / design / tasks
        重点检查：

        边界是不是对的
        是否遗漏隐性约定
        是否把多个问题错误混成一个 change
        tasks.md 是否足够可执行
        第 3 步：必要时废弃 change 重来
        如果 proposal 拆错，不要勉强修补。
        直接重开新的 change 往往更省成本。

        第 4 步：执行 /opsx:apply
        基于已确认的工件实施代码变更。

        第 5 步：跑专项审查
        依次执行：

        /prepare-review
        /spring-architecture-review
        /sql-risk-review
        @"reviewer (agent)"
        第 6 步：执行 /opsx:verify
        确认实现是否和 OpenSpec change 对齐。

        第 7 步：归档
        执行 /opsx:archive，结束当前 change。

        十二、实战流程复盘：一次真实 change 是怎么跑完的
        前面的内容偏方法论整理。
        但如果只讲原则，不保留实践过程，这篇文章会更像“总结”，而不是“复盘”。

        下面把这次 harness 落地中的真实执行过程保留下来，方便团队更直观地理解：一条 change 在项目里到底是怎么从初始化、拆解、修正、执行、审查一路跑完的。

        1. 初始化仓库：先把基础骨架搭起来
        首先通过 openspec init --tools claude 生成 OpenSpec 对应的基础目录结构，然后再补齐 AGENTS.md、CLAUDE.md、REVIEW.md、.claude/ 和 docs/ 这些团队级配置。Pasted image 20260401162722.png这一阶段的目标不是“立刻开始写代码”，而是先把 change 生命周期、项目知识入口、权限和 hook 的承载位置搭好。
        如果这个骨架没立起来，后面的流程就很容易重新滑回“想到哪改到哪”。

        2. 第一轮 propose：先快速成形，再判断是否可执行
        需求进入后，先通过 /opsx:propose 拆解对应 change。Pasted image 20260401162920.pngPasted image 20260331152202.png拆解完成后，可以看到 proposal、design、tasks 等相关文件已经生成：Pasted image 20260331163325.pngPasted image 20260331163653.pngPasted image 20260331163705.png但这里很关键的一点是：第一版拆解并不符合真实需求。

        所以这一步没有继续硬做，而是直接废弃，重新生成新的 change。
        这也是 harness 很重要的一条经验：

        proposal 不是生成出来就要执行，第一版不对，就应该及时推翻。

        3. 第二轮 propose：重新开 change，并持续补充上下文
        第一版废弃之后，重新生成新的 change：Pasted image 20260331164652.png第一次生成后仍然不满足需求，于是继续细化：Pasted image 20260331165333.png

        随后生成新的 proposal：Pasted image 20260331165444.png接着继续把业务逻辑补给 AI，让它修正理解边界：Pasted image 20260331165745.pngPasted image 20260331165841.png这一阶段非常能体现 harness 的现实意义：
        AI 可以很快把 change 搭出一个骨架，但 proposal 的质量，高度依赖你有没有把真实业务边界和上下文交代清楚。

        4. 进入 design：自动生成只是起点，人审才是关键
        在确认了部分逻辑之后，继续生成 design：Pasted image 20260331165958.pngPasted image 20260331173407.png接下来开始人工审计，并修复其中的错误或遗漏：Pasted image 20260331173332.png而且这不是一次就结束，而是多轮审计、持续修正：Pasted image 20260331173524.pngPasted image 20260331173951.pngPasted image 20260331174521.pngPasted image 20260331180512.png这里出现了一个非常典型的问题：
        由于 proposal 中已经把范围限定在 adunit/save、adunit/update，模型始终无法稳定判定到素材相关逻辑。

        所以最后的处理方式不是继续强压一个超大 change，而是：

        当前 change 先只处理当前明确范围内的逻辑
        素材相关逻辑后续再单独开一个 change
        对应生成的 tasks 如下：Pasted image 20260331180902.png这一段过程，其实正好印证了一条非常重要的最佳实践：

        当模型始终无法稳定理解某块逻辑时，优先怀疑的往往不是模型能力，而是 change 边界定义得不够清楚。

        5. verify、apply 和专项审查分开执行
        在任务边界清晰之后，先让子 agent @"reviewer (agent)" 通过 /opsx:verify 做一轮验证：Pasted image 20260331181144.png随后再正式执行 /opsx:apply 开始落代码：Pasted image 20260331185827.png

        代码落地之后，再依次跑专项审查，而不是把所有动作混在一起：

        5.1 /prepare-review：生成 PR 前摘要
        Pasted image 20260331190139.png
        5.2 /spring-architecture-review：进行 Spring 分层架构审计
        Pasted image 20260331190347.png
        5.3 /sql-risk-review：进行数据层和 SQL 风险审计
        Pasted image 20260331190445.png
        5.4 @"reviewer (agent)"：再做一轮只读审计
        Pasted image 20260331190940.png
        5.5 最后再执行 /opsx:verify，检查实现是否仍然和 OpenSpec 工件一致：
        Pasted image 20260331191131.png
        这一步最值得强调的是：
        verify、review、架构审查、SQL 审查不是一回事。

        它们应该明确分工：

        /opsx:verify：检查“实现有没有和 OpenSpec 工件对上”
        /prepare-review：整理“这次到底改了什么，方便人 review”
        /spring-architecture-review：检查“Spring 分层有没有乱”
        /sql-risk-review：检查“SQL 有没有事故味”
        reviewer 子代理：做一轮独立、只读、偏代码审查视角的检查
        之所以不把这些动作一次性打包调用完，不是因为不能这样做，而是因为：

        分开编排更清晰、更可控，也更方便定位到底是哪一个环节出了问题。

        6. 最后归档：一个 change 才算真正结束
        完成验证之后，再执行 /opsx:archive 对当前 change 进行归档：Pasted image 20260401163249.png到了这一步，这个 change 才算从“提出需求”走到了“完成闭环”。

        7. 这段实践过程真正说明了什么
        把这段过程保留下来，比只讲方法论更有价值。因为它更直观地说明了几件事：

        第一版 proposal 往往只是草案，不能盲目执行
        design 阶段的人审，通常比 apply 之后返工便宜得多
        /opsx:verify 不是万能审查，它必须和 review / 架构审查 / SQL 审查分开
        当某块逻辑始终解释不清时，宁可拆 change，也不要把所有问题硬塞进一个 change
        换句话说，harness 的价值不只是“让 AI 参与开发”，而是让整个过程变成：

        可拆解
        可回退
        可审计
        可复用
        可沉淀
        这也是为什么我认为，最佳实践文里应该保留真实过程图，而不只是保留结论。

        十三、Java Spring Boot 项目里最值得加护栏的地方
        并不是所有限制都无脑加越多越好，上下文更应该留给更需要的地方。我始终觉得在限制中更要继续保持简洁，AI才能更好的工作。 比如如果只谈 Java Spring Boot 场景，我认为下面这些坑最值得优先考虑加护栏：

        1. Controller 写业务逻辑
        这是最常见的脏问题。
        所以要在：

        CLAUDE.md
        REVIEW.md
        reviewer
        spring-architecture-review
        里同时卡住。

        2. 直接改 SQL / 配置 / 数据库脚本
        这种改动往往“本地能跑，线上出事”。
        所以应该通过：

        permissions.deny
        guard_write.py
        database.md
        sql-risk-review
        多层保护。

        3. Service 过胖，职责混乱
        这类问题很容易在 AI 连续补代码时越滚越大。
        要靠 reviewer 和架构审查能力持续盯住。

        4. 测试只跑 happy path
        这是很多项目的通病。
        至少要在 testing.md 和 review 摘要里明确：

        跑了哪些测试
        哪些没测
        为什么当前可以接受
        5. 批量更新没有 where 限制
        这类问题不是 bug，而是事故预告。
        数据库规范和 SQL 风险审查里必须把它作为显式检查项。

        十四、来自实战的 3 条额外经验
        除了结构设计本身，这次实践还有 3 条很重要的经验。

        1. 第一版 proposal 通常只是草案
        AI 很容易先产出一个“结构看起来合理、业务边界其实不对”的 proposal。
        因此，proposal 阶段的人审，绝对不能省。

        2. design 阶段修正，比 apply 后返工便宜得多
        一旦进了 apply，错误就开始变成代码、测试和联调成本。
        所以最值得花时间的地方，其实是 design 阶段。

        3. change 边界不清时，宁可拆成多个 change
        如果模型始终无法稳定理解某块逻辑，通常不是因为它“再多想一轮就会懂”，而是因为当前 change 边界定义得不够好。
        这时候最佳实践不是继续硬压，而是：

        把复杂问题拆开，先让 change 边界变清晰。

        十五、给团队的推荐落地顺序
        如果团队想逐步引入这套 harness，我建议按下面顺序落地：

        第一阶段：最小可用版
        先上这几个：

        OpenSpec
        AGENTS.md
        CLAUDE.md
        docs/architecture/implicit-contracts.md
        reviewer
        先把“变更工件化”和“隐性约定显性化”跑通。

        第二阶段：补硬护栏
        再加：

        permissions
        guard_write.py
        ensure_change_context.py
        自动编译 / 测试 / 打包检查
        让高风险动作可控。

        第三阶段：补团队专用能力
        最后再加：

        prepare-review
        spring-architecture-review
        sql-risk-review
        更多团队私有 skills / agents
        让流程真正变成可复用的工程能力。

        十六、附录：可以直接复用的检查清单
        仓库层
        ⬜ 是否有 是否有 `AGENTS.md`
        ⬜ 是否有 是否有 `CLAUDE.md`
        ⬜ 是否有 是否有 `REVIEW.md`
        ⬜ 是否有 是否有 `docs/architecture/implicit-contracts.md`
        ⬜ 是否有 openspec/changes/ 是否有 `openspec/changes/` 目录
        ⬜ 是否有 是否有 `.claude/settings.json`
        ⬜ 是否有 hooks / skills / agents
        流程层
        ⬜ 没有 change 时，不允许直接开始开发
        ⬜ proposal / design / tasks 是否经人工审计
        ⬜ apply 后是否自动跑检查
        ⬜ verify 是否独立执行
        ⬜ change 完成后是否 archive
        风险层
        ⬜ 高风险目录是否已 deny
        ⬜ SQL / Mapper 是否有专项风险检查
        ⬜ 测试缺口是否被显式说明
        ⬜ Spring 分层是否有独立审查
        ⬜ 隐性约定是否在 docs 中显性记录
        如果这些项大部分都能满足，这套 harness 基本就已经具备了在真实项目里稳定运行的基础。

        建议大家根据自己需要去配置，我这次实践的相关配置文件我也已经打包好了，大家可以用于参考，公众号后台回复「20260401」即可获取。

        最后
        说实话，最近 harness 这个词一出来，营销号铺天盖地的水，导致很多人都会有点慌：这又是什么新东西，我是不是又没跟上？
        但它其实没那么吓人。

        因为我们以前做项目的时候，也一直在做差不多的事。我们会定规则，会加限制，会做检查，会留回头路。会告诉工具什么能做，什么不能做；什么可以自动跑，什么一定要人来拍板。
        这些事，我们以前就在做，只是那时候没有一个很统一的名字，把它们都组织到一起。

        所以我想说，别被新词吓到。
        harness 不是突然从天上掉下来的东西。
        它更像是我们早就懂、早就在做的一些工程经验，现在终于被定义出了名字。

        只是到了 AI coding 时代，这件事反而更重要了。
        因为 AI 很像一个特别能干的小帮手，做事快，写东西也快。但它快，不代表它每次都对。
        它能做很多事，也不代表它知道哪条线不能碰。
        所以我们要给它规则，给它边界，给它检查，也给它刹车。
        这其实就是 harness 思想最有价值的地方：不是让 AI 随便冲，而是让它在可控的路上跑。

        我的这篇文章能做的，只是带大家先入门。但后面的路，还是得靠大家自己走。
        因为每个人或者团队做的业务不一样，风险不一样，合作方式也不一样。
        真正适合你的 harness，不会从别人的文章里直接长出来。
        它只能从你们自己的项目里，一点一点试出来，一点一点磨出来。

        说到底，AI 时代不是只会照着别人做就够了。
        会复刻，是第一步。
        但更重要的，是在复刻之后，继续往前走，做出更适合自己的东西。

        我是单向箔，我们下次再见。"""
    
    chunks = chunk_document(doc, {
        "doc": "ai coding guide",
        "lang": "zh",
        "version": "v1"
    })

    for chunk in chunks:
        print(f"============= 分块 {chunk['index']} ===============")
        print(f"metadata: {chunk['metadata']}")
        print(f"content: {chunk['content']}")
        print(f"\n"*5)



if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

import models.qwen.MlpExecution
import models.qwen.TokenInference

namespace Inference.Models.Qwen.DecodeImplementation
open DecodeStage DecodeChain Kernels.Attention TokenInference
set_option maxRecDepth 10000

def StageTrace (impl : MlpExecution.Implementation) : (s : Stage) → Cache s n → Row BF → RopeAngles.Inputs → Type
  | .delta l, c, x, _ => MlpExecution.ResidualTrace impl l.suffix x (l.mixed c (fun _ => x) 0)
  | .attention l, c, x, a => MlpExecution.ResidualTrace impl l.suffix x (l.mixed x a c)

def StageTrace.canonical (impl : MlpExecution.Implementation) : (s : Stage) → (c : Cache s n) → (x : Row BF) →
    (a : RopeAngles.Inputs) → StageTrace impl s c x a
  | .delta l, c, x, _ => impl.canonical l.suffix.mlp (l.suffix.inputBits x (l.mixed c (fun _ => x) 0))
  | .attention l, c, x, a => impl.canonical l.suffix.mlp (l.suffix.inputBits x (l.mixed x a c))

def stageForward (impl : MlpExecution.Implementation) : (s : Stage) → (c : Cache s n) → (x : Row BF) → (a : RopeAngles.Inputs) →
    StageTrace impl s c x a → Row BF
  | .delta l, c, x, _, execution => MlpExecution.residualForward impl l.suffix x (l.mixed c (fun _ => x) 0) execution
  | .attention l, c, x, a, execution => MlpExecution.residualForward impl l.suffix x (l.mixed x a c) execution

theorem stage_eq (impl : MlpExecution.Implementation) (s : Stage) (c : Cache s n) (x : Row BF) (a : RopeAngles.Inputs)
    (execution : StageTrace impl s c x a) : stageForward impl s c x a execution=DecodeStage.forward s c x a := by
  cases s with
  | delta l => exact MlpExecution.residual_eq impl l.suffix x (l.mixed c (fun _ => x) 0) execution
  | attention l => exact MlpExecution.residual_eq impl l.suffix x (l.mixed x a c) execution

def ChainTrace (impl : MlpExecution.Implementation) : (stages : List Stage) → Caches stages n → Row BF → RopeAngles.Inputs → Type
  | [], _, _, _ => PUnit
  | s::rest, c, x, a => (first : StageTrace impl s c.1 x a) × ChainTrace impl rest c.2 (stageForward impl s c.1 x a first) a

def ChainTrace.canonical (impl : MlpExecution.Implementation) : (stages : List Stage) → (c : Caches stages n) → (x : Row BF) →
    (a : RopeAngles.Inputs) → ChainTrace impl stages c x a
  | [], _, _, _ => ⟨⟩
  | s::rest, c, x, a =>
    let first := StageTrace.canonical impl s c.1 x a
    ⟨first, ChainTrace.canonical impl rest c.2 (stageForward impl s c.1 x a first) a⟩

def chain (impl : MlpExecution.Implementation) : (stages : List Stage) → (c : Caches stages n) → (x : Row BF) → (a : RopeAngles.Inputs) →
    ChainTrace impl stages c x a → Row BF × Caches stages (n+1)
  | [], _, x, _, _ => (x, ⟨⟩)
  | s::rest, c, x, a, execution =>
    let tail := chain impl rest c.2 (stageForward impl s c.1 x a execution.1) a execution.2
    (tail.1, (cacheAfter s c.1 x a, tail.2))

theorem chain_eq (impl : MlpExecution.Implementation) (stages : List Stage) (c : Caches stages n) (x : Row BF) (a : RopeAngles.Inputs)
    (execution : ChainTrace impl stages c x a) : chain impl stages c x a execution=DecodeChain.run stages c x a := by
  induction stages generalizing x with
  | nil => rfl
  | cons s rest ih =>
    have tail := ih c.2 (stageForward impl s c.1 x a execution.1) execution.2
    simp only [chain, tail, stage_eq, DecodeChain.run]

abbrev PipelineTrace (impl : MlpExecution.Implementation) (p : Pipeline) (cache : p.graph.Cache n)
    (token : Fin 248320) (angle : RopeAngles.Inputs) := ChainTrace impl p.graph.stages cache (p.embedding.forward token) angle

def pipeline (impl : MlpExecution.Implementation) (p : Pipeline) (cache : p.graph.Cache n) (token : Fin 248320)
    (angle : RopeAngles.Inputs) (execution : PipelineTrace impl p cache token angle) : (Fin 248320 → BF) × p.graph.Cache (n+1) :=
  let result := chain impl p.graph.stages cache (p.embedding.forward token) angle execution
  (p.graph.output.forward (p.graph.finalNorm.forward result.1), result.2)

theorem pipeline_eq (impl : MlpExecution.Implementation) (p : Pipeline) (cache : p.graph.Cache n) (token : Fin 248320)
    (angle : RopeAngles.Inputs) (execution : PipelineTrace impl p cache token angle) : pipeline impl p cache token angle execution=p.run cache token angle := by
  unfold pipeline
  rw [chain_eq]
  rfl

end Inference.Models.Qwen.DecodeImplementation

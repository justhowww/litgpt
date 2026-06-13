# MRT-primary learned-stop FIM diagnostic

This run matches the oracle-length MRT-primary experiment except that:

- candidates generate until EOS or a `2x` target-length safety cap;
- EOS is learned only through MRT visual risk;
- EOS is absent from dataset targets and byte-only CE.

It tests whether visual supervision can jointly learn replacement bytes and
the stopping boundary.

```bash
CLEANUP=0 \
STAGED_CORPUS="$HOME/scratch.metzler-prj/OpenVid-1M_Data/data" \
bash scripts/hpc/zaratan/stage2_fim_psm_mrt_primary_learned_stop_1k_8k/submit_h100.sh
```

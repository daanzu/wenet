
### Training: Custom Model

```bash
python local/combine_short_utts.py -m 5.0 ello-datasets/epic_ello.train90 ello-datasets/epic_ello.train90.csu5.0
python local/combine_short_utts.py -m 5.0 ello-datasets/epic_ello.train10 ello-datasets/epic_ello.train10.csu5.0
python local/combine_short_utts.py -m 5.0 ello-datasets/boulder_learning.train90 ello-datasets/boulder_learning.train90.csu5.0
python local/combine_short_utts.py -m 5.0 ello-datasets/boulder_learning.train10 ello-datasets/boulder_learning.train10.csu5.0

./tools/combine_data.sh data/train_set ello-datasets/{epic_ello.train90,boulder_learning.train90}.csu5.0
./tools/combine_data.sh data/dev_set ello-datasets/{epic_ello.train10,boulder_learning.train10}.csu5.0
./tools/combine_data.sh data/test_set ello-datasets/{epic_ello.test,boulder_learning.test}
./tools/combine_data.sh data/test_set_elb ello-datasets/ello_launch_books.test
./tools/combine_data.sh data/test_set_elb.all ello-datasets/ello_launch_books
./tools/combine_data.sh data/test_set_ee ello-datasets/epic_ello.test
./tools/combine_data.sh data/test_set_bl ello-datasets/boulder_learning.test

bash run.sh --train-set train_set --dev-set dev_set --recog-set "test_set_bl test_set_ee test_set_elb test_set_elb.all" --dir exp/u2pp.blee1 --stage 0 --stop-stage 3
```

### Training: Finetune `gigaspeech_20210728_u2pp_conformer`

```bash
python local/combine_short_utts.py -m 5.0 ello-datasets/epic_ello.train90 ello-datasets/epic_ello.train90.csu5.0
python local/combine_short_utts.py -m 5.0 ello-datasets/epic_ello.train10 ello-datasets/epic_ello.train10.csu5.0

./tools/combine_data.sh data/train_set ello-datasets/epic_ello.train90.csu5.0
./tools/combine_data.sh data/dev_set ello-datasets/epic_ello.train10.csu5.0
./tools/combine_data.sh data/test_set_elb ello-datasets/ello_launch_books.test
./tools/combine_data.sh data/test_set_elb.all ello-datasets/ello_launch_books
./tools/combine_data.sh data/test_set_ee ello-datasets/epic_ello.test
./tools/combine_data.sh data/test_set_bl ello-datasets/boulder_learning.test

bash run_finetune_gigaspeech_old.sh --train-set train_set --dev-set dev_set --recog-set "test_set_elb test_set_elb.all test_set_ee" --dir exp/finetune_ee_csu50_gigaspeech_20210728_u2pp_conformer_exp_1gpu_ep50_lr001_r2 --only-stage 3,4,5
```

### Evaluation

```bash
rg --no-heading --sort path 'Overall -> .* %' exp_dir/(ls -t exp_dir/ | fzf -m --preview 'ls -l exp_dir/{}')/test_*/wer
rg --no-heading --no-filename --no-line-number --sort path 'Epoch \d+ CV info cv_loss' exp_dir/(ls -t exp_dir/ | fzf --preview 'ls -l exp_dir/{}')
```
